export const summaryMetrics = [
  { title: 'Total Spend (Month-to-Date)', value: '$60,452', change: '+3.8%' },
  { title: 'Forecasted Spend', value: '$63,475', change: '-0.5%' },
  { title: 'Active Projects', value: '12' },
  { title: 'Cost Saving Opportunities', value: '4', change: 'New' },
];

export const costBreakdown = [
  { service: 'OpenAI GPT-4', project: 'Project Alpha', spend: '$16,127', usage: '7.5B', trend: '↑', team: 'Core AI', cloudProvider: 'Azure' },
  { service: 'Anthropic Claude 3', project: 'Project Beta', spend: '$26,343', usage: '8.2B', trend: '↔', team: 'Product R&D', cloudProvider: 'AWS' },
  { service: 'Google Gemini Pro', project: 'Project Gamma', spend: '$9,298', usage: '3.1B', trend: '↓', team: 'Data Science', cloudProvider: 'GCP' },
  { service: 'Azure AI Services', project: 'Project Alpha', spend: '$8,684', usage: 'N/A', trend: '↑', team: 'Core AI', cloudProvider: 'Azure' },
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

export const spendOverTimeData = {
  'All Projects': [
    { date: 'Jul 1', spend: 1312 }, { date: 'Jul 2', spend: 1754 }, { date: 'Jul 3', spend: 1512 }, { date: 'Jul 4', spend: 1598 }, { date: 'Jul 5', spend: 1759 }, { date: 'Jul 6', spend: 1841 }, { date: 'Jul 7', spend: 1838 }, { date: 'Jul 8', spend: 1904 }, { date: 'Jul 9', spend: 1888 }, { date: 'Jul 10', spend: 2005 }, { date: 'Jul 11', spend: 1931 }, { date: 'Jul 12', spend: 2041 }, { date: 'Jul 13', spend: 1999 }, { date: 'Jul 14', spend: 2115 }, { date: 'Jul 15', spend: 2163 }, { date: 'Jul 16', spend: 2244 }, { date: 'Jul 17', spend: 2225 }, { date: 'Jul 18', spend: 2383 }, { date: 'Jul 19', spend: 2416 }, { date: 'Jul 20', spend: 2486 }, { date: 'Jul 21', spend: 2453 }, { date: 'Jul 22', spend: 2515 }, { date: 'Jul 23', spend: 2611 }, { date: 'Jul 24', spend: 2623 }, { date: 'Jul 25', spend: 2715 }, { date: 'Jul 26', spend: 2713 }, { date: 'Jul 27', spend: 2804 }, { date: 'Jul 28', spend: 2847 }, { date: 'Jul 29', spend: 2931 }, { date: 'Jul 30', spend: 2913 }, { date: 'Jul 31', spend: 3058 },
  ],
  'Project Alpha': [
    { date: 'Jul 1', spend: 457 }, { date: 'Jul 2', spend: 407 }, { date: 'Jul 3', spend: 401 }, { date: 'Jul 4', spend: 453 }, { date: 'Jul 5', spend: 498 }, { date: 'Jul 6', spend: 561 }, { date: 'Jul 7', spend: 551 }, { date: 'Jul 8', spend: 584 }, { date: 'Jul 9', spend: 578 }, { date: 'Jul 10', spend: 645 }, { date: 'Jul 11', spend: 611 }, { date: 'Jul 12', spend: 681 }, { date: 'Jul 13', spend: 699 }, { date: 'Jul 14', spend: 735 }, { date: 'Jul 15', spend: 763 }, { date: 'Jul 16', spend: 814 }, { date: 'Jul 17', spend: 785 }, { date: 'Jul 18', spend: 883 }, { date: 'Jul 19', spend: 906 }, { date: 'Jul 20', spend: 946 }, { date: 'Jul 21', spend: 893 }, { date: 'Jul 22', spend: 955 }, { date: 'Jul 23', spend: 1011 }, { date: 'Jul 24', spend: 1023 }, { date: 'Jul 25', spend: 1075 }, { date: 'Jul 26', spend: 1033 }, { date: 'Jul 27', spend: 1124 }, { date: 'Jul 28', spend: 1127 }, { date: 'Jul 29', spend: 1191 }, { date: 'Jul 30', spend: 1133 }, { date: 'Jul 31', spend: 1228 },
  ],
  'Project Beta': [
    { date: 'Jul 1', spend: 601 }, { date: 'Jul 2', spend: 1065 }, { date: 'Jul 3', spend: 829 }, { date: 'Jul 4', spend: 812 }, { date: 'Jul 5', spend: 928 }, { date: 'Jul 6', spend: 967 }, { date: 'Jul 7', spend: 954 }, { date: 'Jul 8', spend: 987 }, { date: 'Jul 9', spend: 940 }, { date: 'Jul 10', spend: 1027 }, { date: 'Jul 11', spend: 977 }, { date: 'Jul 12', spend: 1007 }, { date: 'Jul 13', spend: 947 }, { date: 'Jul 14', spend: 1017 }, { date: 'Jul 15', spend: 1047 }, { date: 'Jul 16', spend: 1067 }, { date: 'Jul 17', spend: 1077 }, { date: 'Jul 18', spend: 1127 }, { date: 'Jul 19', spend: 1137 }, { date: 'Jul 20', spend: 1167 }, { date: 'Jul 21', spend: 1177 }, { date: 'Jul 22', spend: 1187 }, { date: 'Jul 23', spend: 1217 }, { date: 'Jul 24', spend: 1217 }, { date: 'Jul 25', spend: 1247 }, { date: 'Jul 26', spend: 1287 }, { date: 'Jul 27', spend: 1297 }, { date: 'Jul 28', spend: 1327 }, { date: 'Jul 29', spend: 1337 }, { date: 'Jul 30', spend: 1377 }, { date: 'Jul 31', spend: 1417 },
  ],
  'Project Gamma': [
    { date: 'Jul 1', spend: 254 }, { date: 'Jul 2', spend: 282 }, { date: 'Jul 3', spend: 282 }, { date: 'Jul 4', spend: 333 }, { date: 'Jul 5', spend: 333 }, { date: 'Jul 6', spend: 313 }, { date: 'Jul 7', spend: 333 }, { date: 'Jul 8', spend: 333 }, { date: 'Jul 9', spend: 370 }, { date: 'Jul 10', spend: 333 }, { date: 'Jul 11', spend: 343 }, { date: 'Jul 12', spend: 353 }, { date: 'Jul 13', spend: 353 }, { date: 'Jul 14', spend: 363 }, { date: 'Jul 15', spend: 353 }, { date: 'Jul 16', spend: 363 }, { date: 'Jul 17', spend: 363 }, { date: 'Jul 18', spend: 373 }, { date: 'Jul 19', spend: 373 }, { date: 'Jul 20', spend: 373 }, { date: 'Jul 21', spend: 383 }, { date: 'Jul 22', spend: 373 }, { date: 'Jul 23', spend: 383 }, { date: 'Jul 24', spend: 383 }, { date: 'Jul 25', spend: 393 }, { date: 'Jul 26', spend: 393 }, { date: 'Jul 27', spend: 383 }, { date: 'Jul 28', spend: 393 }, { date: 'Jul 29', spend: 403 }, { date: 'Jul 30', spend: 403 }, { date: 'Jul 31', spend: 413 },
  ],
};
