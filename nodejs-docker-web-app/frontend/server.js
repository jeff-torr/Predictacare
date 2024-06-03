const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const port = 8080;

app.use(express.static(path.join(__dirname, 'frontend')));
app.use(express.json());

app.post('/run-python', (req, res) => {
    const pythonScriptPath = path.join(__dirname, 'SkLearnNaiveBayes.py');
    const csvFilePath = path.join(__dirname, 'EstimateHealthcareAppointmentLengthGivenX-Sheet1.csv');
    const descriptionField = req.body.descriptionField;

    // Execute the Python script
    exec(`python3 ${pythonScriptPath} "${csvFilePath}" "${descriptionField}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send(error.message);
        }
        console.log(`stdout: ${stdout}`);
        console.error(`stderr: ${stderr}`);
        res.send({ output: stdout.trim() });
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
