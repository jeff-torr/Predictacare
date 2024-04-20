import express from "express";
import dotenv from "dotenv";
dotenv.config();

const PORT = process.env.PORT || 8080;
const app = express();

app.get("/", (req, res) => {
    res.json({ message: " We have mounted the volume to running container" });
});

app.listen(PORT, () => {
    console.log(`App running on ${PORT}`);
});