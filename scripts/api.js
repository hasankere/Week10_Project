import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const fetchPrices = async () => {
    const response = await axios.get(`${API_URL}/prices`);
    return response.data;
};

export const fetchEvents = async () => {
    const response = await axios.get(`${API_URL}/events`);
    return response.data;
};

export const fetchMetrics = async () => {
    const response = await axios.get(`${API_URL}/metrics`);
    return response.data;
};
