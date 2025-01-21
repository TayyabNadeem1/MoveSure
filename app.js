const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const bodyParser = require('body-parser'); 
const app = express();

const cors = require('cors');
app.use(cors());

// Middleware to serve static files from 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.json()); // Parse JSON bodies

//Connect to MongoDB (make sure MongoDB is running)
mongoose.connect('mongodb://localhost:27017/movesure', { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('MongoDB connection error:', err));

//Define a User Schema
const userSchema = new mongoose.Schema({
    email: {
        type: String,
        required: true, // Make sure email is required
    },
    password: {
        type: String,
        required: true, // Make sure password is required
    },
});

//Create a User Model
const User = mongoose.model('User', userSchema);

//Define the Register Route
app.post('/register', async (req, res) => {
    console.log('Request Body:', req.body); // Log the request body

    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).send('Email and password are required');
    }

    try {
  
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: 'User already exists' });
        }

        // Save the new user to the database (no password hashing)
        const newUser = new User({ email, password });
        await newUser.save();

        res.status(200).json({ message: 'User registered successfully' });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: 'Server error' });
    }
});


//Login Route
app.post('/login', async (req, res) => {
    console.log("Login route hit");
    console.log("Request Body:", req.body);

    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ message: 'Email and password are required' });
    }

    try {

        const user = await User.findOne({ email, password });
        if (user) {
            return res.status(200).json({ message: 'Login successful' });
        } else {
            return res.status(401).json({ message: 'Invalid email or password' });
        }
    } catch (err) {
        console.error('Error during login:', err);
        res.status(500).json({ message: 'Server error' });
    }
});


// Serve the register.html and login.html pages
app.get('/register', (req, res) => {
    res.sendFile(path.join(__dirname, 'register.html'));
});

app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, 'login.html'));
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
