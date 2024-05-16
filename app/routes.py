from flask import Flask, render_template, request, redirect, jsonify
from neural_network import util_functions
from neural_network import solver
from neural_network import problem
from neural_network import deep_network

import torch
from torch import nn

app = Flask(__name__)

network = deep_network.Approximator(num_neurons=32, num_layers=4)
optimizer = torch.optim.Adam(network.parameters(), lr = 0.005)
answer = solver.HeatSolver(optimizer=optimizer, network=network, loss=nn.MSELoss())

# ------------------------------
@app.route("/", methods=["POST", 'GET'])
def get_input():

    if request.method == 'POST':

        form_data = util_functions.get_attr_dictionary(request.form)
        print(form_data)
        try:
            task = problem.Heat.from_string_dict(form_data)
            answer.task = task
            print('hey')
        except ValueError as e:
            return render_template('input.html', error_presence=True,
                                   error_text = str(e), saved_attributes = form_data)
        print(task)
        
        answer.fit(answer.task)
        answer.plot("E:/KPI\Diploma/app/static/images/plot")
        return redirect('/output')


    return render_template('input.html')

@app.get("/output")
def get_output():

    print(answer.task)


    answer_print_pars = answer.task.to_latex()
    
    return render_template('output.html', path='images/plot.png', 
                           a = answer_print_pars['coefficient'],
                           init_boundary =  answer_print_pars['init-boundary'],
                           left_boundary =  answer_print_pars['left-boundary'],
                           right_boundary =  answer_print_pars['right-boundary'])

@app.post("/output")
def get_data():

    print(request.form)
    try:
        x = float(request.form['x'])
        t = float(request.form['t'])
    #     answer.task = task
    #     print('hey')
    except ValueError as e:
        return render_template('output.html', error_presence=True,
                               error_text = str(e))

    # answer.plot("E:/KPI\Diploma/app/static/images/plot.png")
    
    answer_print_pars = answer.task.to_latex()

    return render_template('output.html', path='images/plot.png', 
                           a = answer_print_pars['coefficient'],
                           init_boundary =  answer_print_pars['init-boundary'],
                           left_boundary =  answer_print_pars['left-boundary'],
                           right_boundary =  answer_print_pars['right-boundary'])


@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()
    x = data['x_input']
    t = data['t_input']
    try:
        x = torch.tensor(float(x))
        t = torch.tensor(float(t))
    except ValueError:
        y = "This are not numbers at all. shit over youself."
        return jsonify(y)
    
    y = answer.predict(t, x)

    return f"{y.item():.6f}"
    # return str(y.item())