import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(message)s]'
)

project_name = "depth-estimation"

list_of_files = [

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/settings.py",
    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/model/base_estimator.py",
    f"src/{project_name}/model/midas_estimator.py",
    f"src/{project_name}/model/depth_anything_estimator.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/preprocessor.py",
    f"src/{project_name}/pipeline/postprocessor.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/device.py",

    "app/__init__.py",
    "app/gradio_app.py",
    "app/components/__init__.py",
    "app/components/ui_components.py",

    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_preprocessor.py",
    "tests/test_postprocessor.py",
    "tests/test_model_factory.py",
    "tests/test_postprocessor.py",

    "notebooks/exploration.ipynb",
    "examples/sample_batch.csv",
    "scripts/batch_depth.py",

    ".github/workflows/ci.yaml",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt"

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")

