/*
@author: Max Rivera
*/
const backendURL = "http://127.0.0.1:5000/backend";
var heating_output = document.getElementById("heating_load");
var cooling_output = document.getElementById("cooling_load");


// predict button
const predictBtn = document.querySelector('#predict_button');
predictBtn.addEventListener('click',function() {

    // grab all the inputs from html 
    relative_compactness = document.getElementById('relative_compactness').value;
    surface_area = document.getElementById('surface_area').value;
    wall_area = document.getElementById('wall_area').value;
    roof_area = document.getElementById('roof_area').value;
    overall_height = document.getElementById('overall_height').value;
    orientation_ = document.getElementById('orientation').value;
    glazing_area = document.getElementById('glazing_area').value;
    glazing_area_distribution = document.getElementById('glazing_area_distribution').value;

    // create a new FormData object and append input elements 
    let formData = new FormData();
    formData.append("relative_compactness",relative_compactness);
    formData.append("surface_area", surface_area);
    formData.append("wall_area",wall_area);
    formData.append("roof_area",roof_area);
    formData.append("overall_height",overall_height);
    formData.append("orientation",orientation_);
    formData.append("glazing_area", glazing_area);
    formData.append("glazing_area_distribution",glazing_area_distribution);
    console.log("hello");


    // REST API call
    // let response = await fetch(backendURL, {
    //     method: "POST",
    //     body: formData
    // });

    // if (response.ok) {
    //     alert("fetch worked");
    // } else {
    //     alert("HTTP-Error: " + response.status);
    // }

    fetch(backendURL, {
        method:"POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            heating_output.value = data['message'];
            cooling_output.value = data['message'];
        });

    


});