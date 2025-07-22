"""
Financial Metrics Calculator Module

This module provides core financial calculation utilities for real estate investments.
Keeps calculations separate from the main engine for better modularity.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """Container for financial analysis results."""
    roi: float
    irr: float
    npv: float
    cash_on_cash_return: float
    cap_rate: float
    debt_service_coverage_ratio: float
    break_even_occupancy: float
    payback_period: int


class FinancialCalculator:
    """Core financial calculations for real estate investments."""
    
    @staticmethod
    def calculate_mortgage_payment(loan_amount: float, annual_rate: float, term_years: int) -> float:
        """Calculate monthly mortgage payment."""
        if loan_amount <= 0 or term_years <= 0:
            return 0.0
        
        monthly_rate = annual_rate / 100 / 12
        num_payments = term_years * 12
        
        if monthly_rate == 0:
            return loan_amount / num_payments
        
        payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
                 ((1 + monthly_rate) ** num_payments - 1)
        
        return payment
    
    @staticmethod
    def calculate_irr(cash_flows: List[float], max_iterations: int = 100) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        try:
            # Initial guess
            rate = 0.1
            
            for _ in range(max_iterations):
                npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                npv_derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
                
                if abs(npv_derivative) < 1e-10:
                    break
                
                new_rate = rate - npv / npv_derivative
                
                if abs(new_rate - rate) < 1e-10:
                    break
                
                rate = new_rate
            
            return rate
            
        except (ZeroDivisionError, OverflowError):
            return 0.0
    
    @staticmethod
    def calculate_npv(initial_investment: float, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = -initial_investment
        for i, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** (i + 1))
        return npv
    
    @staticmethod
    def calculate_cap_rate(noi: float, property_value: float) -> float:
        """Calculate Capitalization Rate."""
        if property_value <= 0:
            return 0.0
        return (noi / property_value) * 100
    
    @staticmethod
    def calculate_cash_on_cash_return(annual_cash_flow: float, total_cash_invested: float) -> float:
        """Calculate Cash-on-Cash Return."""
        if total_cash_invested <= 0:
            return 0.0
        return (annual_cash_flow / total_cash_invested) * 100
    
    @staticmethod
    def calculate_dscr(noi: float, debt_service: float) -> float:
        """Calculate Debt Service Coverage Ratio."""
        if debt_service <= 0:
            return 0.0
        return noi / debt_service