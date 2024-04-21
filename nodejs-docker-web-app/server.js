import express from "express";
import dotenv from "dotenv";
import { fileURLToPath } from 'url'; // Import fileURLToPath function from the 'url' module
import path from "path";

dotenv.config();

const PORT = process.env.PORT || 8080;
const app = express();

// Define __dirname using import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve static files from the 'frontend' directory
app.use(express.static(path.join(__dirname, 'frontend')));

// Serve the UI (HTML page) for the root route
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`App running on ${PORT}`);
});