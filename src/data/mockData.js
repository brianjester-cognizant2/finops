export const summaryMetrics = [
  { title: 'Total Spend (Month-to-Date)', value: '$82,800', change: '+5.2%' },
  { title: 'Forecasted Spend', value: '$145,000', change: '-1.5%' },
  { title: 'Active Projects', value: '12' },
  { title: 'Cost Saving Opportunities', value: '4', change: 'New' },
];

export const costBreakdown = [
  { service: 'OpenAI GPT-4', project: 'Project Alpha', spend: '$29,925', usage: '8.0B', trend: '↑' },
  { service: 'Anthropic Claude 3', project: 'Project Beta', spend: '$21,280', usage: '6.5B', trend: '↓' },
  { service: 'Google Gemini Pro', project: 'Project Gamma', spend: '$18,620', usage: '9.9B', trend: '↑' },
  { service: 'Azure AI Services', project: 'Project Alpha', spend: '$12,975', usage: 'N/A', trend: '↔' },
];

export const recentAnomalies = [
  { timestamp: '2024-07-21 10:45 UTC', service: 'OpenAI GPT-4', description: 'Sudden 300% spike in API calls from dev environment.', severity: 'High' },
  { timestamp: '2024-07-20 18:00 UTC', service: 'Google Gemini Pro', description: 'Usage cost exceeded daily budget by 20%.', severity: 'Medium' },
  { timestamp: '2024-07-20 09:12 UTC', service: 'Anthropic Claude 3', description: 'High latency detected for prompt evaluation.', severity: 'Low' },
];

export const placeholderPages = {
  costAnalysis: {
    title: 'Cost Analysis',
    description: 'This page will contain detailed cost analysis tools, including spend across services, projects, teams, and regions. It will allow for deep dives into cost distribution and trend identification.',
  },
  monitoring: {
    title: 'Monitoring & Alerts',
    description: 'This page will feature continuous monitoring dashboards, alert configuration, and logs. It will be used to track metrics, detect anomalies, and manage cost control strategies like prompt monitoring.',
  },
  recommendations: {
    title: 'Recommendations',
    description: 'This page will provide actionable insights and recommendations for cost savings. This includes right-sizing instances, suggesting better subscription plans, and highlighting best practices for effective cost allocation.',
  },
};

export const spendOverTime = [
  { date: 'Jul 1', spend: 2200 },
  { date: 'Jul 2', spend: 2500 },
  { date: 'Jul 3', spend: 2300 },
  { date: 'Jul 4', spend: 2800 },
  { date: 'Jul 5', spend: 3100 },
  { date: 'Jul 6', spend: 2900 },
  { date: 'Jul 7', spend: 3400 },
  { date: 'Jul 8', spend: 3500 },
  { date: 'Jul 9', spend: 3800 },
  { date: 'Jul 10', spend: 4000 },
  { date: 'Jul 11', spend: 4100 },
  { date: 'Jul 12', spend: 4300 },
  { date: 'Jul 13', spend: 4200 },
  { date: 'Jul 14', spend: 4500 },
  { date: 'Jul 15', spend: 4400 },
  { date: 'Jul 16', spend: 4700 },
  { date: 'Jul 17', spend: 5000 },
  { date: 'Jul 18', spend: 5200 },
  { date: 'Jul 19', spend: 5100 },
  { date: 'Jul 20', spend: 5300 },
  { date: 'Jul 21', spend: 5500 },
  { date: 'Jul 22', spend: 5600 },
  { date: 'Jul 23', spend: 5800 },
  { date: 'Jul 24', spend: 5700 },
  { date: 'Jul 25', spend: 6000 },
  { date: 'Jul 26', spend: 6200 },
  { date: 'Jul 27', spend: 6100 },
  { date: 'Jul 28', spend: 6400 },
  { date: 'Jul 29', spend: 6500 },
  { date: 'Jul 30', spend: 6700 },
  { date: 'Jul 31', spend: 7000 },
];
