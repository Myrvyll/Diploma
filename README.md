## About project
This is an implementation of PINN (Physics-informed neural network) for solving 1d heat equation. 
It was used for the bachelor's thesis as programming component of the task.
It is created using:
+ PyTorch for neural network,
+ numpy for calculations
+ html/css/js/bootstrap5 for front-end
+ flask for web-server

## Getting started

First of all, it is needed to set up an environment with pip.
```
pip install -r requirements.txt
```

Then to launch a project you have to load a flask server.
```
flask --app app/routes run
```

This gives you an interface, running on http://127.0.0.1:5000.
After example input, some time will pass while NN is learning a solution. Logs of the learning process will be present in the console.

## Test example

Given problem can be used for demonstration purposes.

![зображення](https://github.com/user-attachments/assets/e7a3d944-0ff5-4d37-9af4-52d897939711)

