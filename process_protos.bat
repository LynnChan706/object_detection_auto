for %%c in (object_detection/protos/*.proto ) do  protoc object_detection/protos/%%c --python_out=. 
pause