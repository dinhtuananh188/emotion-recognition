from roboflow import Roboflow
rf = Roboflow(api_key="ivVCWsQujwpr5z4jwGxE")
project = rf.workspace("detect-55qxx").project("facial-expression-by-gemini")
version = project.version(2)
dataset = version.download("yolov11")
                