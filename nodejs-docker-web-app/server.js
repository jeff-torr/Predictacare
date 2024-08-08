import express from "express";
import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import path from "path";
import { exec } from "child_process";

dotenv.config();

const PORT = process.env.PORT || 8080;
const app = express();

// Define __dirname using import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve static files from the 'frontend' directory
app.use(express.static(path.join(__dirname, 'frontend')));
app.use(express.json());

// Serve the UI (HTML page) for the root route
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

// Handle favicon requests
app.get('/favicon.ico', (req, res) => res.status(204));

// Define the /run-python POST route
app.post('/run-python', (req, res) => {
    const pythonScriptPath = path.join(__dirname, 'SkLearnLinearRegression.py');
    const csvFilePath = path.join(__dirname, 'EstimateHealthcareAppointmentLengthGivenX-Sheet2.csv');
    const descriptionField = req.body.descriptionField;
    const ageField = req.body.ageField;
    const genderField = req.body.genderField;
    const familiarityField = req.body.familiarityField;

    // Execute the Python script with the provided description field
    const command = `python3 ${pythonScriptPath} ${csvFilePath} "${descriptionField}" ${ageField} "${genderField}" ${familiarityField}`;
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send(error.message);
        }
        
        // Log the standard output from the Python script
        console.log(`stdout: ${stdout}`);
        
        // Log any errors from the Python script
        if (stderr) {
            console.error(`stderr: ${stderr}`);
        }
        
        // Send the output back to the client
        res.send({ output: stdout.trim() });
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`App running on ${PORT}`);
});
