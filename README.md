# AIconomy

This project is an example of reinforcement learning applied to the global economy. It uses AI and reinforcement learning to analyze the potential factors that could impact the global economy and make predictions about the future. The model uses historical macroeconomic and microeconomic data, as well as the current political landscape, to simulate the global economy and make predictions about the future. The AI and reinforcement learning system then adjusts the variables in the model in order to achieve the desired outcome.

This project is written in Python and uses the NumPy, Pandas, and Keras libraries. The code consists of an EconomyEnv class that is used to create the environment, an Agent class that is used to create the agent, and a train function that is used to train the agent. 

This project is intended to be used as an example of how to apply AI and reinforcement learning to the global economy. It is not intended to be a complete solution, but rather a starting point for further exploration and development.


Macroeconomic variables include factors like GDP, inflation, unemployment, and interest rates. These variables are heavily intertwined and any change in one of these variables can have a significant impact on the global economy. 

1. Gross Domestic Product (GDP): GDP = Consumption (C) + Investment (I) + Government Spending (G) + Net Exports (X - M)
2. National Income Equation: National Income (NI) = GDP - Depreciation
3. Aggregate Demand (AD): AD = Consumption (C) + Investment (I) + Government Spending (G) + Net Exports (X – M)
4. Aggregate Supply (AS): AS = Output (Y) = Aggregate Supply Curve Function (A) + (1 - b)Y
5. Money Supply (MS): MS = Central Bank Money (CB) + M1 + M2
6. Consumption Function: C = Autonomous Consumption (a) + bYd
7. Investment Function: I = Investment Autonomous Component (I0) + I1Y
8. Government Expenditure Function: G = Government Autonomous Expenditure (G0) + G1Y
9. Net Exports Function: NX = Net Exports Autonomous Component (NX0) + NX1Y
10. Phillips Curve: Inflation (πt) = πt-1 + β(unemployment - natural rate of unemployment (u*))
11. Output Gap: Output (Y) - Potential Output (Y*) = Yt - Yt*
12. Balance of Payments (BOP): BOP = Consumption (C) + Investment (I) + Government Spending (G) + Net Exports (NX)
13. Exchange Rate: ER = Exchange Rate at Time t (ERt) / Exchange Rate at Time t-1 (ERt-1)
14. Interest Rate: r = Interest Rate (i) + Inflation (πt)
15. Fiscal Multiplier: Change in Output (ΔY) = Fiscal Multiplier (γ) * Change in Government Spending (ΔG)
16. Monetary Multiplier: Change in Money Supply (ΔM) = Change in Consumption (ΔC) / Change in M1
17. IS Curve: Output (Y) = Potential Output (Y*) + (1/α)(Interest Rate (r) - Natural Interest Rate (r*))
18. LM Curve: Interest Rate (r) = Natural Interest Rate (r*) - α(Output (Y) - Potential Output (Y*))
19. Natural Rate of Unemployment: Unemployment (u*) = Unemployment (u) + φ(Output (Y) - Potential Output (Y*))

Microeconomic variables include the behavior of firms, households, and other agents in the economy. These variables can include the decisions of businesses to invest, the decisions of households to consume, and the decisions of governments to intervene. 

1. Demand and Supply: This formula measures the relationship between the demand for a good or service and its corresponding supply. It is used to determine the equilibrium price and quantity in the market. 
2. Production Possibility Curve (PPC): This formula measures the maximum output of two goods or services that can be produced given the available resources. It is used to illustrate the trade-offs between the two goods and services and the opportunity cost associated with producing more of one good or service over another.
3. Marginal Cost Curve: This formula measures the cost associated with producing one more unit of output. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
4. Marginal Revenue Curve: This formula measures the additional revenue associated with producing one more unit of output. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
5. Average Cost Curve: This formula measures the average cost associated with producing a given amount of output. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
6. Average Variable Cost Curve: This formula measures the average variable cost associated with producing a given amount of output. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
7. Average Fixed Cost Curve: This formula measures the average fixed cost associated with producing a given amount of output. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
8. Law of Diminishing Marginal Utility: This formula measures the decrease in utility associated with consuming one more unit of a good or service. It is used to explain why consumers are willing to pay less for additional units of a good or service.
9. Price Elasticity of Demand: This formula measures the responsiveness of quantity demanded to changes in price. It is used to determine the optimal price for a good or service.
10. Price Elasticity of Supply: This formula measures the responsiveness of quantity supplied to changes in price. It is used to determine the optimal price for a good or service.
11. Total Cost Curve: This formula measures the total cost associated with producing a given amount of output. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
12. Short-Run Cost Curve: This formula measures the cost associated with producing a given amount of output in the short-run. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
13. Long-Run Cost Curve: This formula measures the cost associated with producing a given amount of output in the long-run. It is used to determine the optimal level of output for firms and to formulate pricing strategies.
14. Monopoly Pricing: This formula measures the pricing strategy of a monopoly firm. It is used to determine the optimal price and quantity of output for a monopoly firm in order to maximize profit.
15. Perfect Competition Pricing: This formula measures the pricing strategy of a perfectly competitive firm. It is used to determine the optimal price and quantity of output for a perfectly competitive firm in order to maximize profit.
16. Cournot Duopoly: This formula measures the pricing strategy of a duopoly. It is used to determine the optimal price and quantity of output for a duopoly in order to maximize profit.
17. Stackelberg Duopoly: This formula measures the pricing strategy of a duopoly. It is used to determine the optimal price and quantity of output for a duopoly in order to maximize profit.
18. Nash Equilibrium: This formula measures the optimal strategy for two or more players in a game. It is used to determine the optimal strategy for players in a game in order to maximize the expected payoff.

Political variables involve the policies of governments and any changes in the political landscape that could affect the economy. These policies can include taxation, trade restrictions, and other regulations. 

1. Government Spending: This is the total amount of money spent by the government on public services and infrastructure. This affects the economic growth of Portugal, as it can increase the demand for goods and services, and create new jobs.
2. Taxation: Taxes are a major source of revenue for the government. They can be used to fund public services and infrastructure, or to reduce the national debt. The level of taxation affects the economic growth of Portugal, as higher taxes can reduce consumer spending, investment and business expansion.
3. Monetary Policy: The Central Bank of Portugal implements monetary policy, which is the manipulation of interest rates and the creation of money. This affects the economic growth of Portugal, as it can influence the cost of borrowing and the availability of credit.
4. Exchange Rate: The exchange rate is the rate at which one currency is exchanged for another. This affects the economic growth of Portugal, as it can influence the cost of imports and exports.
5. Fiscal Policy: This is the use of taxes and government spending to influence the economy. It can be used to encourage investment, consumption, and economic growth.
6. Trade Agreements: Portugal has trade agreements with other countries, which can affect the economic growth of Portugal. Trade agreements can reduce tariffs and other trade barriers, which can enable Portuguese companies to access foreign markets.
7. Foreign Investment: Foreign investment can be an important source of capital for businesses in Portugal. This can create jobs and stimulate economic growth.
8. Labour Market Policies: These are the policies that the government puts in place to regulate the labour market. These policies can affect the economic growth of Portugal, as they can influence the cost of labour and the availability of workers.
