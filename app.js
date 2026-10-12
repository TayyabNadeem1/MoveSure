require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');
const bcrypt = require('bcryptjs');

const app = express();

app.use(cors());
app.use(express.static(path.join(__dirname, 'static')));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Connect to MongoDB using environment variable
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/movesure';
mongoose.connect(MONGO_URI)
    .then(() => console.log('[MoveSure Node] Connected to MongoDB'))
    .catch(err => console.warn('[MoveSure Node] MongoDB connection notice:', err.message));

// User Schema with hashed passwords
const userSchema = new mongoose.Schema({
    email: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        lowercase: true,
    },
    password: {
        type: String,
        required: true,
    },
    createdAt: {
        type: Date,
        default: Date.now,
    }
});

const User = mongoose.model('User', userSchema);

// Register Route
app.post('/register', async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ status: 'error', message: 'Email and password are required' });
    }

    try {
        const existingUser = await User.findOne({ email: email.toLowerCase() });
        if (existingUser) {
            return res.status(400).json({ status: 'error', message: 'User already exists' });
        }

        // Hash password securely with bcrypt
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        const newUser = new User({ email: email.toLowerCase(), password: hashedPassword });
        await newUser.save();

        res.status(200).json({ status: 'success', message: 'User registered successfully' });
    } catch (err) {
        console.error('Registration error:', err);
        res.status(500).json({ status: 'error', message: 'Server error during registration' });
    }
});

// Login Route
app.post('/login', async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ status: 'error', message: 'Email and password are required' });
    }

    try {
        const user = await User.findOne({ email: email.toLowerCase() });
        if (!user) {
            return res.status(401).json({ status: 'error', message: 'Invalid email or password' });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(401).json({ status: 'error', message: 'Invalid email or password' });
        }

        res.status(200).json({ status: 'success', message: 'Login successful' });
    } catch (err) {
        console.error('Login error:', err);
        res.status(500).json({ status: 'error', message: 'Server error during login' });
    }
});

// Static HTML fallbacks
app.get('/register', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'register.html'));
});

app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'login.html'));
});

app.get('/home', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'home.html'));
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`[MoveSure Node] Server is running on http://localhost:${port}`);
});
