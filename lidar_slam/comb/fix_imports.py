import re
import os

def fix_type_hints_in_file(filepath):
    if not os.path.exists(filepath):
        return
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix type hints by adding quotes
    patterns = [
        (r': PoseEstimate([^a-zA-Z_])', r': "PoseEstimate"\1'),
        (r': FeatureDescriptor([^a-zA-Z_])', r': "FeatureDescriptor"\1'),
        (r': AssociationScore([^a-zA-Z_])', r': "AssociationScore"\1'),
        (r': ValidationResult([^a-zA-Z_])', r': "ValidationResult"\1'),
        (r'-> PoseEstimate([^a-zA-Z_])', r'-> "PoseEstimate"\1'),
        (r'Optional\[PoseEstimate\]', r'Optional["PoseEstimate"]'),
        (r'List\[PoseEstimate\]', r'List["PoseEstimate"]'),
        (r'List\[AssociationScore\]', r'List["AssociationScore"]'),
        (r'List\[FeatureDescriptor\]', r'List["FeatureDescriptor"]'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed: {filepath}")

# Fix all files
for file in ['feature_association.py', 'association_validator.py', 
             'hybrid_pose_estimator.py', 'association_visualizer.py']:
    fix_type_hints_in_file(file)