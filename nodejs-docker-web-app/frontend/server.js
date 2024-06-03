const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const port = 8080;

// Serve static files from the "frontend" directory
app.use(express.static(path.join(__dirname, 'frontend')));
app.use(express.json());

// Handle favicon requests
app.get('/favicon.ico', (req, res) => res.status(204));

// Define the /run-python POST route
app.post('/run-python', (req, res) => {
    const pythonScriptPath = path.join(__dirname, 'SkLearnNaiveBayes.py');
    const csvFilePath = path.join(__dirname, 'EstimateHealthcareAppointmentLengthGivenX-Sheet1.csv');
    const descriptionField = req.body.descriptionField;

    // Execute the Python script with the provided description field
    exec(`python3 ${pythonScriptPath} "${csvFilePath}" "${descriptionField}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send(error.message);
        }
        
        // Log the standard output from the Python script
        console.log(`stdout: ${stdout}`);
        
        // Log any errors from the Python script
        console.error(`stderr: ${stderr}`);
        
        // Send the output back to the client
        res.send({ output: stdout.trim() });
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
