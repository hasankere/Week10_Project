import React,{ useEffect, useState } from 'react';
import ChartComponent from './ChartComponent';
import EventImpact from './EventImpact';
import FilterPanel from './FilterPanel';
import MetricsDisplay from './MetricsDisplay';
import { fetchPrices, fetchEvents, fetchMetrics } from '../services/api';

const Dashboard = () => {
    const [prices, setPrices] = useState([]);
    const [events, setEvents] = useState([]);
    const [metrics, setMetrics] = useState({});

    useEffect(() => {
        fetchPrices().then(data => setPrices(data));
        fetchEvents().then(data => setEvents(data));
        fetchMetrics().then(data => setMetrics(data));
    }, []);

    return (
        <div>
            <h1>Brent Oil Price Dashboard</h1>
            <FilterPanel />
            <ChartComponent data={prices} />
            <EventImpact events={events} />
            <MetricsDisplay metrics={metrics} />
        </div>
    );
};

export default Dashboard;
