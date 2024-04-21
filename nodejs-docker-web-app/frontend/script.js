document.addEventListener('DOMContentLoaded', function() {
    console.log('Script loaded successfully!');
});



//depending on age, gender, etc. multiply the minute value that comes back from bakcend by a scalar before returning it to user

function getClassData() {
    var nameField = document.getElementById('nameField').value;
    var ageField = document.gerElementByID('ageField').value;
    var genderField = document.gerElementByID('genderField').value;
    var desriptionField = document.gerElementByID('descriptionField').value;
    var familiarityField = document.gerElementByID('familiarityField').value;
}

var subButton = document.getElementById('subButton');
subButton.addEventListener('click', getClassData, false); 
// submits form (calls above function) when button clicked

class UserData100 {
    constructor(name, age, gender, description, familiarity) {
      this.name = nameField;
      this.age = ageField;
      this.gender = genderFIeld;
      this.description = descriptionField;
      this.familiarity = familiarityField;
    }
  }