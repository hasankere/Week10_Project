import React from 'react';
import { Line } from 'react-chartjs-2';

const ChartComponent = ({ data }) => {
    const chartData = {
        labels: data.map(item => item.Date),
        datasets: [{
            label: 'Brent Oil Price (USD)',
            data: data.map(item => item.Price),
            borderColor: 'blue',
            fill: false,
        }]
    };

    return (
        <div>
            <h2>Price Trend Over Time</h2>
            <Line data={chartData} />
        </div>
    );
};

export default ChartComponent;
