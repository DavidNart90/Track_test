"""
Financial Analytics and Market Intelligence Engine for TrackRealties AI Platform

This module provides comprehensive financial analysis capabilities including ROI calculations,
cash flow analysis, risk assessment, and market trend analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal

from ..models.financial import InvestmentParams, ROIProjection, CashFlowAnalysis, RiskAssessment
from ..models.market import MarketDataPoint
from ..models.property import PropertyListing
from ..core.config import get_settings
from .financial_metrics import FinancialCalculator
from .market_intelligence import MarketIntelligenceEngine, MarketTrend
from .cma_engine import ComparativeMarketAnalysis

logger = logging.getLogger(__name__)
settings = get_settings()


class FinancialAnalyticsEngine:
    """
    Core financial analytics engine for real estate investment analysis.
    Provides ROI calculations, cash flow analysis, and market intelligence.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculator = FinancialCalculator()
        self.market_engine = MarketIntelligenceEngine()
        self.cma_engine = ComparativeMarketAnalysis()
    
    def calculate_roi_projection(self, params: InvestmentParams,
                               market_data: Optional[List[MarketDataPoint]] = None) -> ROIProjection:
        """Calculate comprehensive ROI projection with multiple scenarios."""
        try:
            # Base calculations
            purchase_price = params.purchase_price
            down_payment = params.down_payment_amount  # Use property method
            loan_amount = params.loan_amount  # Use property method
            total_investment = params.total_initial_investment  # Use property method
            
            # Calculate monthly mortgage payment
            monthly_payment = self.calculator.calculate_mortgage_payment(
                loan_amount, params.loan_interest_rate, params.loan_term_years
            )
            
            # Annual income and expenses
            annual_rental_income = params.monthly_rent * 12
            effective_rental_income = params.effective_monthly_rent * 12  # Use property method
            
            annual_expenses = (
                params.property_tax_annual +
                params.insurance_annual +
                (purchase_price * Decimal(str(params.maintenance_percent))) +  # Calculate maintenance
                (params.utilities_monthly * 12) +
                (params.hoa_monthly * 12) +
                (effective_rental_income * params.property_management_percent / 100)
            )
            
            # Net Operating Income
            noi = effective_rental_income - annual_expenses
            annual_debt_service = monthly_payment * 12
            annual_cash_flow = noi - annual_debt_service
            
            # Key metrics
            cap_rate = self.calculator.calculate_cap_rate(noi, purchase_price)
            cash_on_cash_return = self.calculator.calculate_cash_on_cash_return(annual_cash_flow, total_investment)
            
            # Multi-year projections
            projections = self._calculate_multi_year_projections(
                params, noi, annual_cash_flow, purchase_price, loan_amount
            )
            
            # Calculate IRR
            cash_flows = [-total_investment] + [proj['cash_flow'] for proj in projections[:5]]
            irr = self.calculator.calculate_irr(cash_flows) * 100
            
            return ROIProjection(
                initial_investment=total_investment,
                annual_cash_flow=annual_cash_flow,
                cap_rate=cap_rate,
                cash_on_cash_return=cash_on_cash_return,
                irr=irr,
                five_year_roi=(projections[4]['total_return'] / total_investment) * 100,
                projections=projections
            )
            
        except Exception as e:
            self.logger.error(f"ROI calculation failed: {e}")
            raise
    
    def analyze_cash_flow(self, params: InvestmentParams) -> CashFlowAnalysis:
        """Perform detailed cash flow analysis."""
        try:
            # Calculate all components
            purchase_price = params.purchase_price
            down_payment = params.down_payment_amount  # Use property method
            loan_amount = params.loan_amount  # Use property method
            
            # Monthly mortgage payment
            monthly_mortgage = self.calculator.calculate_mortgage_payment(
                loan_amount, params.loan_interest_rate, params.loan_term_years
            )
            
            # Monthly income
            monthly_rental_income = params.monthly_rent
            effective_monthly_income = params.effective_monthly_rent  # Use property method
            
            # Monthly expenses
            monthly_property_tax = params.property_tax_annual / 12
            monthly_insurance = params.insurance_annual / 12
            monthly_maintenance = purchase_price * Decimal(str(params.maintenance_percent)) / 12
            monthly_management = (effective_monthly_income * params.property_management_percent / 100)
            
            total_monthly_expenses = (
                monthly_property_tax +
                monthly_insurance +
                monthly_maintenance +
                params.utilities_monthly +
                params.hoa_monthly +
                monthly_management
            )
            
            # Net cash flow
            monthly_noi = effective_monthly_income - total_monthly_expenses
            monthly_cash_flow = monthly_noi - monthly_mortgage
            
            return CashFlowAnalysis(
                monthly_rental_income=monthly_rental_income,
                effective_monthly_income=effective_monthly_income,
                monthly_expenses=total_monthly_expenses,
                monthly_mortgage_payment=monthly_mortgage,
                monthly_noi=monthly_noi,
                monthly_cash_flow=monthly_cash_flow,
                annual_cash_flow=monthly_cash_flow * 12,
                expense_breakdown={
                    'property_tax': monthly_property_tax,
                    'insurance': monthly_insurance,
                    'maintenance': monthly_maintenance,
                    'utilities': params.utilities_monthly,
                    'hoa': params.hoa_monthly,
                    'management': monthly_management
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cash flow analysis failed: {e}")
            raise
    
    def assess_investment_risk(self, params: InvestmentParams,
                             market_data: Optional[List[MarketDataPoint]] = None) -> RiskAssessment:
        """Assess investment risk factors."""
        try:
            risk_factors = []
            risk_score = 0  # 0-100 scale, higher = riskier
            
            # Leverage risk
            ltv_ratio = (1 - params.down_payment_percent / 100) * 100
            if ltv_ratio > 80:
                risk_factors.append("High leverage (LTV > 80%)")
                risk_score += 15
            elif ltv_ratio > 90:
                risk_factors.append("Very high leverage (LTV > 90%)")
                risk_score += 25
            
            # Cash flow risk
            cash_flow_analysis = self.analyze_cash_flow(params)
            if cash_flow_analysis.monthly_cash_flow < 0:
                risk_factors.append("Negative cash flow")
                risk_score += 30
            elif cash_flow_analysis.monthly_cash_flow < 200:
                risk_factors.append("Minimal cash flow buffer")
                risk_score += 15
            
            # Vacancy risk
            if params.vacancy_rate > 10:
                risk_factors.append("High vacancy assumption (>10%)")
                risk_score += 10
            
            # Market risk (if market data available)
            market_volatility = None
            if market_data:
                market_volatility = self.market_engine.calculate_market_volatility(market_data)
                if market_volatility > 0.15:
                    risk_factors.append("High market volatility")
                    risk_score += 20
            
            # Determine risk level
            if risk_score <= 20:
                risk_level = "Low"
            elif risk_score <= 40:
                risk_level = "Moderate"
            elif risk_score <= 60:
                risk_level = "High"
            else:
                risk_level = "Very High"
            
            return RiskAssessment(
                risk_level=risk_level,
                risk_score=risk_score,
                risk_factors=risk_factors,
                leverage_ratio=ltv_ratio,
                cash_flow_risk=cash_flow_analysis.monthly_cash_flow < 0,
                market_volatility=market_volatility
            )
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            raise
    
    def analyze_market_trends(self, market_data: List[MarketDataPoint],
                            timeframe_days: int = 90) -> MarketTrend:
        """Analyze market trends from historical data."""
        return self.market_engine.analyze_market_trends(market_data, timeframe_days)
    
    def forecast_property_value(self, current_value: float, market_data: List[MarketDataPoint],
                              forecast_months: int = 12) -> dict:
        """Forecast future property value based on market trends."""
        return self.market_engine.forecast_property_value(current_value, market_data, forecast_months)
    
    def _calculate_multi_year_projections(self, params: InvestmentParams,
                                        initial_noi: float, initial_cash_flow: float,
                                        purchase_price: float, loan_amount: float) -> List[Dict[str, Any]]:
        """Calculate multi-year financial projections."""
        projections = []
        current_noi = initial_noi
        current_property_value = purchase_price
        remaining_loan = loan_amount
        
        annual_mortgage = self.calculator.calculate_mortgage_payment(
            loan_amount, params.loan_interest_rate, params.loan_term_years
        ) * 12
        
        for year in range(1, 11):  # 10-year projection
            # Apply appreciation and rent growth
            current_property_value *= (1 + params.property_appreciation_rate / 100)
            current_noi *= (1 + params.annual_rent_increase / 100)
            
            # Calculate principal paydown
            interest_payment = remaining_loan * params.loan_interest_rate / 100
            principal_payment = annual_mortgage - interest_payment
            remaining_loan = max(0, remaining_loan - principal_payment)
            
            # Calculate cash flow and equity
            annual_cash_flow = current_noi - annual_mortgage
            current_equity = current_property_value - remaining_loan
            
            # Total return (cash flow + equity gain)
            equity_gain = current_property_value - purchase_price
            total_return = (year * annual_cash_flow) + equity_gain
            
            projections.append({
                'year': year,
                'property_value': round(current_property_value, 2),
                'noi': round(current_noi, 2),
                'cash_flow': round(annual_cash_flow, 2),
                'equity': round(current_equity, 2),
                'total_return': round(total_return, 2),
                'remaining_loan': round(remaining_loan, 2)
            })
        
        return projections