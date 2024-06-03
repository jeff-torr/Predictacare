document.addEventListener('DOMContentLoaded', function() {
    console.log('Script loaded successfully!');
});



//depending on age, gender, etc. multiply the minute value that comes back from bakcend by a scalar before returning it to user

function getClassData() {
    var nameField = document.getElementById('nameField').value;
    var ageField = document.getElementById('ageField').value;
    var genderField = document.getElementById('genderField').value;
    var desriptionField = document.getElementById('descriptionField').value;
    var familiarityField = document.getElementById('familiarityField').value;
}

var subButton = document.getElementById('subButton');
subButton.addEventListener('click', getClassData, false); 
// submits form (calls above function) when button clicked

class UserData100 {
    constructor(name, age, gender, description, familiarity) {
      this.name = nameField;
      this.age = ageField;
      this.gender = genderField;
      this.description = descriptionField;
      this.familiarity = familiarityField;
    }
  }

//need to create a shell drop
//run python SkLearn ... in the cmdline with the descriptionField as the argument
//alter naive bayes to create file with number of minutes in it and get it back into frontend
//delte file after read

//free hosting with website on it? cloudrun is cheap. school might have free domains

//free website