document.addEventListener('DOMContentLoaded', function() {
  console.log('Script loaded successfully!');
  
  var subButton = document.getElementById('subButton');
  subButton.addEventListener('click', getClassData, false);
});

function getClassData() {
  var nameField = document.getElementById('nameField').value;
  var ageField = document.getElementById('ageField').value;
  var genderField = document.getElementById('genderField').value;
  var descriptionField = document.getElementById('descriptionField').value;
  var familiarityField = document.getElementById('familiarityField').value;

  const userData = {
      nameField,
      ageField,
      genderField,
      descriptionField,
      familiarityField
  };

  fetch('/run-python', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(userData)
  })
  .then(response => response.json())
  .then(data => {
      let minutes = parseFloat(data.output);

    //   // Determine age multiplier
    //   let ageMultiplier = 1.0;
    //   if (ageField < 13) {
    //     ageMultiplier = 1.1
    //   } else if (ageField < 30) {
    //       ageMultiplier = 1.0;
    //   } else if (ageField < 50) {
    //       ageMultiplier = 1.1;
    //   } else if (ageField < 70) {
    //       ageMultiplier = 1.2;
    //   } else {
    //       ageMultiplier = 1.3;
    //   }

    //   // Determine gender multiplier
    //   let genderMultiplier = genderField === 'female' ? 1.1 : 1.0;

    //   // Determine familiarity multiplier
    //   let familiarityMultiplier = familiarityField === 'yes' ? 0.9 : 1.1;

    //   // Calculate adjusted minutes
    //   let adjustedMinutes = minutes * ageMultiplier * genderMultiplier * familiarityMultiplier;

      document.getElementById('result').innerText = `Estimated Appointment Length: ${minutes.toFixed(2)} minutes`;
  })
  .catch(error => console.error('Error:', error));
}
