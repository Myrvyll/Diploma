from flask import Flask, render_template, request, redirect
import neural_network.util_functions as util_functions
from ..neural_network import solver
from ..neural_network import problem

app = Flask(__name__)


# ------------------------------
@app.route("/", methods=["POST", 'GET'])
def get_input():

    if request.method == 'POST':

        form_data = util_functions.get_attr_dictionary(request.form)
        # print(form_data)
        try:
            task = problem.Heat.from_string_dict(form_data)
        except ValueError as e:
            return render_template('input.html', error_presence=True,
                                   error_text = str(e))
        # return redirect('/output')

    return render_template('input.html')

@app.route("/output", methods=["POST", 'GET'])
def post_output():

    if request.method == 'POST':
        return redirect('/')
    return render_template('output.html')

