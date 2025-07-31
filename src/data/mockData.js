export const summaryMetrics = [
  { title: 'Total Spend (Month-to-Date)', value: '$12,450', change: '+5.2%' },
  { title: 'Forecasted Spend', value: '$15,800', change: '-1.5%' },
  { title: 'Active Projects', value: '12' },
  { title: 'Cost Saving Opportunities', value: '4', change: 'New' },
];

export const costBreakdown = [
  { service: 'OpenAI GPT-4', project: 'Project Alpha', spend: '$4,500', usage: '1.2B', trend: '↑' },
  { service: 'Anthropic Claude 3', project: 'Project Beta', spend: '$3,200', usage: '980M', trend: '↓' },
  { service: 'Google Gemini Pro', project: 'Project Gamma', spend: '$2,800', usage: '1.5B', trend: '↑' },
  { service: 'Azure AI Services', project: 'Project Alpha', spend: '$1,950', usage: 'N/A', trend: '↔' },
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