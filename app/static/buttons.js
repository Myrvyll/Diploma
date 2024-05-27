document.getElementById("calculation_form_button").addEventListener("click", function () {
    var t_input = document.getElementById("t_input").value;
    var x_input = document.getElementById("x_input").value;

    var data = {
      "t_input": t_input,
      "x_input": x_input,
    };

    fetch("/process_data", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.text())
      .then((result) => {
        document.getElementById("nn_value").innerHTML = "Значення: " + result;
      })
      .catch((error) => {
        alert("Error occurred while processing data.");
        console.error("Error:", error);
      });
  });


  document.getElementById("toggle_right").addEventListener("click", function () {

      var illustrationNN = document.getElementById("illustration_nn");
      var illustrationFDM = document.getElementById("illustration_fdm");
      var illustrationSol = document.getElementById("illustration_sol");

      var imageUrlNN = illustrationNN.src;
      var imageUrlFDM = illustrationFDM.src;
      var imageUrlSol = illustrationSol.src;
      var number = parseInt(imageUrlNN.match(/(\d+)(?=\.png)/));

      var angleChange = 60
   
      if (number < (180-angleChange)) {
          number += angleChange;
          var newImageUrlNN = imageUrlNN.replace(/(\d+)(?=\.png)/, number);
          var newImageUrlFDM = imageUrlFDM.replace(/(\d+)(?=\.png)/, number);
          var newImageUrlSol = imageUrlSol.replace(/(\d+)(?=\.png)/, number);
          illustrationNN.setAttribute('src', newImageUrlNN);
          illustrationFDM.setAttribute('src', newImageUrlFDM);
          illustrationSol.setAttribute('src', newImageUrlSol);
      };
  });

  document.getElementById("toggle_left").addEventListener("click", function () {

      var illustrationNN = document.getElementById("illustration_nn");
      var illustrationFDM = document.getElementById("illustration_fdm");
      var illustrationSol = document.getElementById("illustration_sol");

      var imageUrlNN = illustrationNN.src;
      var imageUrlFDM = illustrationFDM.src;
      var imageUrlSol = illustrationSol.src;
      var number = parseInt(imageUrlNN.match(/(\d+)(?=\.png)/));

      var angleChange = 60
   
      if (number > angleChange) {
          number -= angleChange;
          var newImageUrlNN = imageUrlNN.replace(/(\d+)(?=\.png)/, number);
          var newImageUrlFDM = imageUrlFDM.replace(/(\d+)(?=\.png)/, number);
          var newImageUrlSol = imageUrlSol.replace(/(\d+)(?=\.png)/, number);
          illustrationNN.setAttribute('src', newImageUrlNN);
          illustrationFDM.setAttribute('src', newImageUrlFDM);
          illustrationSol.setAttribute('src', newImageUrlSol);
      };
  });

