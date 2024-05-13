from flask import Flask, render_template, request, redirect
from .. import neural_network.util_functions

app = Flask(__name__)





# ------------------------------
@app.route("/", methods=["POST", 'GET'])
def get_input():

    if request.method == 'POST':

        form_data = .get_attr_dictionary(request.form)
        print(form_data)
        # return redirect('/output')

    return render_template('input.html')

@app.route("/output", methods=["POST", 'GET'])
def post_output():

    if request.method == 'POST':
        return redirect('/')
    return render_template('output.html')

