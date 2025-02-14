import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class OccupancyGrid:
    def __init__(self, mapXLength, mapYLength, initXY, unitGridSize, lidarFOV, numSamplesPerRev, lidarMaxRange, wallThickness):
        xNum = int(mapXLength / unitGridSize)
        yNum = int(mapYLength / unitGridSize)
        x = np.linspace(-xNum * unitGridSize / 2, xNum * unitGridSize / 2, num=xNum + 1) + initXY['x']
        y = np.linspace(-xNum * unitGridSize / 2, xNum * unitGridSize / 2, num=yNum + 1) + initXY['y']
        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(x, y)
        self.occupancyGridVisited = np.ones((xNum + 1, yNum + 1))
        self.occupancyGridTotal = 2 * np.ones((xNum + 1, yNum + 1))
        self.unitGridSize = unitGridSize
        self.lidarFOV = lidarFOV
        self.lidarMaxRange = lidarMaxRange
        self.wallThickness = wallThickness
        self.mapXLim = [self.OccupancyGridX[0, 0], self.OccupancyGridX[0, -1]]
        self.mapYLim = [self.OccupancyGridY[0, 0], self.OccupancyGridY[-1, 0]]
        self.numSamplesPerRev = numSamplesPerRev
        self.angularStep = lidarFOV / numSamplesPerRev
        self.numSpokes = int(np.rint(2 * np.pi / self.angularStep))
        xGrid, yGrid, bearingIdxGrid, rangeIdxGrid = self.spokesGrid()
        radByX, radByY, radByR = self.itemizeSpokesGrid(xGrid, yGrid, bearingIdxGrid, rangeIdxGrid)
        self.radByX = radByX
        self.radByY = radByY
        self.radByR = radByR
        self.spokesStartIdx = int(((self.numSpokes / 2 - self.numSamplesPerRev) / 2) % self.numSpokes)
        
        # Initialize plot
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.trajectory_x = []
        self.trajectory_y = []

    def updateOccupancyGrid(self, reading, dTheta=0, update=True):
        x, y, theta, rMeasure = reading['x'], reading['y'], reading['theta'], reading['range']
        
        # Update trajectory
        self.trajectory_x.append(x)
        self.trajectory_y.append(y)
        
        theta += dTheta
        rMeasure = np.asarray(rMeasure)
        spokesOffsetIdxByTheta = int(np.rint(theta / (2 * np.pi) * self.numSpokes))
        emptyXList, emptyYList, occupiedXList, occupiedYList = [], [], [], []
        
        for i in range(self.numSamplesPerRev):
            spokeIdx = int(np.rint((self.spokesStartIdx + spokesOffsetIdxByTheta + i) % self.numSpokes))
            xAtSpokeDir = self.radByX[spokeIdx]
            yAtSpokeDir = self.radByY[spokeIdx]
            rAtSpokeDir = self.radByR[spokeIdx]
            
            if rMeasure[i] < self.lidarMaxRange:
                emptyIdx = np.argwhere(rAtSpokeDir < rMeasure[i] - self.wallThickness / 2)
            else:
                emptyIdx = []
            
            occupiedIdx = np.argwhere(
                (rAtSpokeDir > rMeasure[i] - self.wallThickness / 2) & 
                (rAtSpokeDir < rMeasure[i] + self.wallThickness / 2))
            
            xEmptyIdx, yEmptyIdx = self.convertRealXYToMapIdx(
                x + xAtSpokeDir[emptyIdx], y + yAtSpokeDir[emptyIdx])
            xOccupiedIdx, yOccupiedIdx = self.convertRealXYToMapIdx(
                x + xAtSpokeDir[occupiedIdx], y + yAtSpokeDir[occupiedIdx])
            
            if update:
                self.checkAndExapndOG(
                    x + xAtSpokeDir[occupiedIdx], y + yAtSpokeDir[occupiedIdx])
                if len(emptyIdx) != 0:
                    self.occupancyGridTotal[yEmptyIdx, xEmptyIdx] += 1
                if len(occupiedIdx) != 0:
                    self.occupancyGridVisited[yOccupiedIdx, xOccupiedIdx] += 2
                    self.occupancyGridTotal[yOccupiedIdx, xOccupiedIdx] += 2
            else:
                emptyXList.extend(x + xAtSpokeDir[emptyIdx])
                emptyYList.extend(y + yAtSpokeDir[emptyIdx])
                occupiedXList.extend(x + xAtSpokeDir[occupiedIdx])
                occupiedYList.extend(y + yAtSpokeDir[occupiedIdx])
        
        # Update plot after each grid update
        self.plot_current_state()
        
        if not update:
            return (np.asarray(emptyXList), np.asarray(emptyYList), 
                   np.asarray(occupiedXList), np.asarray(occupiedYList))

    def plot_current_state(self, xRange=None, yRange=None):
        """Plot the current state of the occupancy grid and trajectory"""
        self.ax.clear()
        
        if xRange is None or xRange[0] < self.mapXLim[0] or xRange[1] > self.mapXLim[1]:
            xRange = self.mapXLim
        if yRange is None or yRange[0] < self.mapYLim[0] or yRange[1] > self.mapYLim[1]:
            yRange = self.mapYLim
            
        ogMap = self.occupancyGridVisited / self.occupancyGridTotal
        xIdx, yIdx = self.convertRealXYToMapIdx(xRange, yRange)
        ogMap = ogMap[yIdx[0]: yIdx[1], xIdx[0]: xIdx[1]]
        ogMap = np.flipud(1 - ogMap)
        
        # Plot occupancy grid
        self.ax.imshow(ogMap, cmap='gray', 
                      extent=[xRange[0], xRange[1], yRange[0], yRange[1]])
        
        # Plot trajectory
        if len(self.trajectory_x) > 0:
            self.ax.plot(self.trajectory_x, self.trajectory_y, 'b-', linewidth=1)
            self.ax.scatter(self.trajectory_x[-1], self.trajectory_y[-1], 
                          color='red', s=100, label='Current Position')
            if len(self.trajectory_x) > 1:
                self.ax.scatter(self.trajectory_x[0], self.trajectory_y[0], 
                              color='green', s=100, label='Start Position')
        
        self.ax.set_title(f'Step {len(self.trajectory_x)}')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Short pause to allow the plot to update

    # [Rest of the methods remain the same as in the original OccupancyGrid class]
    def spokesGrid(self):
        numHalfElem = int(self.lidarMaxRange / self.unitGridSize)
        bearingIdxGrid = np.zeros((2 * numHalfElem + 1, 2 * numHalfElem + 1))
        x = np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 2 * numHalfElem + 1)
        y = np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 2 * numHalfElem + 1)
        xGrid, yGrid = np.meshgrid(x, y)
        bearingIdxGrid[:, numHalfElem + 1: 2 * numHalfElem + 1] = np.rint((np.pi / 2 + np.arctan(
            yGrid[:, numHalfElem + 1: 2 * numHalfElem + 1] / xGrid[:, numHalfElem + 1: 2 * numHalfElem + 1]))
                / np.pi / 2 * self.numSpokes - 0.5).astype(int)
        bearingIdxGrid[:, 0: numHalfElem] = np.fliplr(np.flipud(bearingIdxGrid))[:, 0: numHalfElem] + int(self.numSpokes / 2)
        bearingIdxGrid[numHalfElem + 1: 2 * numHalfElem + 1, numHalfElem] = int(self.numSpokes / 2)
        rangeIdxGrid = np.sqrt(xGrid**2 + yGrid**2)
        return xGrid, yGrid, bearingIdxGrid, rangeIdxGrid

    def itemizeSpokesGrid(self, xGrid, yGrid, bearingIdxGrid, rangeIdxGrid):
        radByX = []
        radByY = []
        radByR = []
        for i in range(self.numSpokes):
            idx = np.argwhere(bearingIdxGrid == i)
            radByX.append(xGrid[idx[:, 0], idx[:, 1]])
            radByY.append(yGrid[idx[:, 0], idx[:, 1]])
            radByR.append(rangeIdxGrid[idx[:, 0], idx[:, 1]])
        return radByX, radByY, radByR

    def convertRealXYToMapIdx(self, x, y):
        xIdx = (np.rint((x - self.mapXLim[0]) / self.unitGridSize)).astype(int)
        yIdx = (np.rint((y - self.mapYLim[0]) / self.unitGridSize)).astype(int)
        return xIdx, yIdx

    def checkMapToExpand(self, x, y):
        if any(x < self.mapXLim[0]):
            return 1
        elif any(x > self.mapXLim[1]):
            return 2
        elif any(y < self.mapYLim[0]):
            return 3
        elif any(y > self.mapYLim[1]):
            return 4
        else:
            return -1

    def checkAndExapndOG(self, x, y):
        expandDirection = self.checkMapToExpand(x, y)
        while (expandDirection != -1):
            self.expandOccupancyGrid(expandDirection)
            expandDirection = self.checkMapToExpand(x, y)

    def expandOccupancyGridHelper(self, position, axis):
        gridShape = self.occupancyGridVisited.shape
        if axis == 0:
            insertion = np.ones((int(gridShape[0] / 5),  gridShape[1]))
            if position == 0:
                x = self.OccupancyGridX[0]
                y = np.linspace(self.mapYLim[0] - int(gridShape[0] / 5) * self.unitGridSize, self.mapYLim[0],
                                num=int(gridShape[0] / 5), endpoint=False)
            else:
                x = self.OccupancyGridX[0]
                y = np.linspace(self.mapYLim[1] + self.unitGridSize, self.mapYLim[1] + (int(gridShape[0] / 5) ) * self.unitGridSize,
                                num=int(gridShape[0] / 5), endpoint=False)
        else:
            insertion = np.ones((gridShape[0], int(gridShape[1] / 5)))
            if position == 0:
                y = self.OccupancyGridY[:, 0]
                x = np.linspace(self.mapXLim[0] - int(gridShape[1] / 5) * self.unitGridSize, self.mapXLim[0],
                                num=int(gridShape[1] / 5), endpoint=False)
            else:
                y = self.OccupancyGridY[:, 0]
                x = np.linspace(self.mapXLim[1] + self.unitGridSize, self.mapXLim[1] + (int(gridShape[1] / 5)) * self.unitGridSize,
                                num=int(gridShape[1] / 5), endpoint=False)
        self.occupancyGridVisited = np.insert(self.occupancyGridVisited, [position], insertion, axis=axis)
        self.occupancyGridTotal = np.insert(self.occupancyGridTotal, [position], 2 * insertion, axis=axis)
        xv, yv = np.meshgrid(x, y)
        self.OccupancyGridX = np.insert(self.OccupancyGridX, [position], xv, axis=axis)
        self.OccupancyGridY = np.insert(self.OccupancyGridY, [position], yv, axis=axis)
        self.mapXLim[0] = self.OccupancyGridX[0, 0]
        self.mapXLim[1] = self.OccupancyGridX[0, -1]
        self.mapYLim[0] = self.OccupancyGridY[0, 0]
        self.mapYLim[1] = self.OccupancyGridY[-1, 0]

    def expandOccupancyGrid(self, expandDirection):
        gridShape = self.occupancyGridVisited.shape
        if expandDirection == 1:
            self.expandOccupancyGridHelper(0, 1)
        elif expandDirection == 2:
            self.expandOccupancyGridHelper(gridShape[1], 1)
        elif expandDirection == 3:
            self.expandOccupancyGridHelper(0, 0)
        else:
            self.expandOccupancyGridHelper(gridShape[0], 0)

def main():
    # Initialize parameters
    initMapXLength = 10  # meters
    initMapYLength = 10  # meters
    unitGridSize = 0.02  # meters
    lidarFOV = np.pi  # radians
    lidarMaxRange = 10  # meters
    wallThickness = 7 * unitGridSize

    # Load sensor data
    jsonFile = "../DataSet/DataPreprocessed/intel-gfs"
    with open(jsonFile, 'r') as f:
        input = json.load(f)
        sensorData = input['map']

    # Get number of samples per revolution from first reading
    numSamplesPerRev = len(sensorData[list(sensorData)[0]]['range'])
    
    # Get initial position
    initXY = sensorData[sorted(sensorData.keys())[0]]

    # Initialize OccupancyGrid
    og = OccupancyGrid(initMapXLength, initMapYLength, initXY, unitGridSize, 
                       lidarFOV, numSamplesPerRev, lidarMaxRange, wallThickness)

    # Process each sensor reading
    count = 0
    for key in sorted(sensorData.keys()):
        count += 1
        print(f"Processing reading {count}")
        
        # Update occupancy grid with current reading
        og.updateOccupancyGrid(sensorData[key])
        
        # Optional: break after certain number of readings for testing
        #if count == 100:
        #    break

    # Keep the final plot window open
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()