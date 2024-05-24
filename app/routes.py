from flask import Flask, render_template, request, redirect, jsonify, session
from flask_session import Session
import dill
import torch
from torch import nn


from neural_network import util_functions
from neural_network import solver_nn
from neural_network import solver_fdm
from neural_network import problem
from neural_network import deep_network

# torch.manual_seed(42)

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


network = deep_network.Approximator(num_neurons=128, num_layers=4)
optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)


# ------------------------------

@app.route("/", methods=["POST", 'GET'])
def get_input():

    nn_answer = solver_nn.NNHeatSolver(optimizer=optimizer, network=network, loss=nn.MSELoss())

    if request.method == 'POST':
        form_data = util_functions.get_attr_dictionary(request.form)
        print(form_data)
        try:
            task = problem.Heat.from_string_dict(form_data)
            nn_answer.task = task
            fdm_answer = solver_fdm.FDMHeatSolver(task)

        except ValueError as e:
            return render_template('input.html', error_presence=True,
                                   error_text=str(e), saved_attributes=form_data)
        print(task)
        # Neural network solution
        nn_answer.fit(nn_answer.task, epochs=500)
        nn_answer.plot("E:/KPI/Diploma/app/static/images/plot_nn")
        nn_answer.plot_loss("E:/KPI/Diploma/app/static/images/plot_loss")

        # FDM solution
        fdm_answer.solve(t_n=32, x_n=32)
        fdm_answer.plot("E:/KPI/Diploma/app/static/images/plot_fdm")
        nn_answer.task.plot_solution("E:/KPI/Diploma/app/static/images/plot_sol")

        # Save the nn_answer in the session
        ser_nn = dill.dumps(nn_answer)
        session['nn_answer'] = ser_nn
        # Save the fdm_answer in the session
        ser_fdm = dill.dumps(fdm_answer)
        session['fdm_answer'] = ser_fdm


        return redirect('/output')

    return render_template('input.html')

@app.get("/output")
def get_output():
    # Load nn_answer and fdm_answer from the session
    nn_answer = dill.loads(session.get('nn_answer'))
    fdm_answer = dill.loads(session.get('fdm_answer'))

    # Check if the deserialization was successful
    if nn_answer is None or fdm_answer is None:
        return "Error: Failed to retrieve solutions."

    error_nn = nn_answer.count_metrics(1000)
    error_fdm = fdm_answer.count_metrics(1000)
    print(error_nn)
    print(error_fdm)

    answer_print_pars = nn_answer.task.to_latex()

    return render_template('output.html', path='images/plot.png',
                           a=answer_print_pars['coefficient'],
                           init_boundary=answer_print_pars['init-boundary'],
                           left_boundary=answer_print_pars['left-boundary'],
                           right_boundary=answer_print_pars['right-boundary'],
                           error_nn=error_nn,
                           error_fdm=error_fdm)

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

    # Load nn_answer and fdm_answer from the session
    nn_answer = dill.loads(session.get('nn_answer'))

    # Check if the deserialization was successful
    if nn_answer is None:
        return "Error: Failed to retrieve solutions."
    
    answer_print_pars = nn_answer.task.to_latex()

    return render_template('output.html', path='images/plot.png', 
                           a = answer_print_pars['coefficient'],
                           init_boundary =  answer_print_pars['init-boundary'],
                           left_boundary =  answer_print_pars['left-boundary'],
                           right_boundary =  answer_print_pars['right-boundary'])


@app.route('/process_data', methods=['POST'])
def process_data():
    # Load nn_answer and fdm_answer from the session
    nn_answer = dill.loads(session.get('nn_answer'))

    # Check if the deserialization was successful
    if nn_answer is None:
        return "Error: Failed to retrieve solutions."

    data = request.get_json()
    x = data['x_input']
    t = data['t_input']
    try:
        x = torch.tensor(float(x))
        t = torch.tensor(float(t))
    except ValueError:
        y = "This are not numbers at all. Please, correct your input."
        return jsonify(y)
    
    y = nn_answer.predict(t, x)

    return f"{y.item():.6f}"
    # return str(y.item())

if __name__ == "__main__":
    app.run()