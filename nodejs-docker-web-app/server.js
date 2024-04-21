import express from "express";
import dotenv from "dotenv";
dotenv.config();

const PORT = process.env.PORT || 8080;
const app = express();

app.get("/", (req, res) => {
    res.json({ message: " We have mounted the volume to running container" });
});

// Serve static files from the 'frontend' directory
// app.use(express.static(path.join(__dirname, 'frontend')));

// app.get("/", (req, res) => {
//     res.sendFile(path.join(__dirname, 'frontend', 'index.html')); // Send the HTML file
// });

app.listen(PORT, () => {
    console.log(`App running on ${PORT}`);
});