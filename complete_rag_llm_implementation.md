# Complete RAG + Real Estate LLM Implementation Guide
## End-to-End Intelligent Agent for TrackRealties

### 🏗️ **System Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT AGENT LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  Role-Specific Agents: Investor | Developer | Buyer | Agent     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ Investment  │ │ Development │ │ Home Buying │ │ Agent Biz   ││
│  │ Advisor     │ │ Advisor     │ │ Assistant   │ │ Intelligence││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  Smart Router → Query Analysis → Strategy Selection → Execution │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ Intent      │ │ Entity      │ │ Context     │ │ Tool        ││
│  │ Classifier  │ │ Extractor   │ │ Manager     │ │ Selector    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Vector    │ │    Graph    │ │   Hybrid    │ │ Hallucination│
│  │   Search    │ │   Search    │ │   Search    │ │  Detection  ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ Vector DB   │ │ Knowledge   │ │ Real Estate │ │ External    ││
│  │ (PGVector)  │ │ Graph       │ │ Market LLM  │ │ APIs        ││
│  │             │ │ (Neo4j)     │ │ (Fine-tuned)│ │ (Zillow/MLS)││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## **Phase 1: Foundation Components (Weeks 1-4)**

### **1.1 Real Estate Market LLM Development**

#### **A. Data Preparation for Fine-Tuning**

```python
# data_preparation.py
import pandas as pd
import json
from typing import List, Dict, Any

class RealEstateDataPreprocessor:
    """
    Prepare real estate data for LLM fine-tuning
    """
    
    def __init__(self):
        self.role_templates = {
            "investor": self._create_investor_examples,
            "developer": self._create_developer_examples,
            "buyer": self._create_buyer_examples,
            "agent": self._create_agent_examples
        }
    
    def prepare_training_data(self, market_data_df: pd.DataFrame, 
                            property_data: List[Dict]) -> List[Dict[str, str]]:
        """
        Convert raw real estate data into training examples
        """
        training_examples = []
        
        # Process market data
        for _, market_row in market_data_df.iterrows():
            for role in self.role_templates:
                examples = self.role_templates[role](market_row, property_data)
                training_examples.extend(examples)
        
        return training_examples
    
    def _create_investor_examples(self, market_row: pd.Series, 
                                property_data: List[Dict]) -> List[Dict[str, str]]:
        """Create training examples for investor role"""
        region = market_row['REGION_NAME']
        median_price = market_row['MEDIAN_SALE_PRICE']
        price_yoy = market_row['MEDIAN_SALE_PRICE_YOY']
        dom = market_row['MEDIAN_DAYS_ON_MARKET']
        
        examples = [
            {
                "instruction": f"Analyze the investment potential for {region}",
                "input": f"I'm looking to invest in {region}. What's the market like?",
                "output": f"""🎯 **Investment Snapshot**
{region} shows {'strong' if price_yoy > 0.05 else 'moderate'} growth potential with {price_yoy*100:+.1f}% YoY price appreciation. Current median price is ${median_price:,.0f} with properties moving in {dom} days.

📊 **Market Analysis**
- Median Sale Price: ${median_price:,.0f} ({price_yoy*100:+.1f}% YoY)
- Market Velocity: {dom} days on market
- Market Trend: {'Appreciating' if price_yoy > 0 else 'Declining'}

💰 **Investment Strategy**
{'Focus on cash flow properties in this appreciating market' if price_yoy > 0.05 else 'Look for value plays and distressed opportunities'}

📋 **Next Steps**
1. Analyze specific neighborhoods within {region}
2. Calculate cap rates for target properties
3. Connect with local property management companies"""
            },
            
            {
                "instruction": f"Calculate ROI for rental property in {region}",
                "input": f"What ROI can I expect from rental properties in {region}?",
                "output": f"""💰 **ROI Analysis for {region}**

Based on current market conditions:
- Purchase Price Range: ${median_price*0.8:,.0f} - ${median_price*1.2:,.0f}
- Estimated Rental Yield: {self._estimate_rental_yield(median_price):.1f}%
- Cash-on-Cash Return: {self._estimate_cash_on_cash(median_price):.1f}%

🎯 **Investment Metrics**
- Cap Rate: {self._estimate_cap_rate(median_price):.1f}%
- Appreciation Potential: {price_yoy*100:+.1f}% based on current trends
- Market Risk: {'Low' if dom < 60 else 'Medium' if dom < 100 else 'High'}

📋 **Action Items**
1. Target properties under ${median_price*0.9:,.0f}
2. Factor in property taxes and insurance
3. Analyze rental demand by neighborhood"""
            }
        ]
        
        return examples
    
    def _create_developer_examples(self, market_row: pd.Series, 
                                 property_data: List[Dict]) -> List[Dict[str, str]]:
        """Create training examples for developer role"""
        region = market_row['REGION_NAME']
        active_listings = market_row['ACTIVE_LISTINGS']
        months_supply = market_row['MONTHS_OF_SUPPLY']
        
        examples = [
            {
                "instruction": f"Assess development opportunity in {region}",
                "input": f"Is {region} good for residential development?",
                "output": f"""🏗️ **Development Overview**
{region} shows {'strong' if months_supply < 3 else 'moderate'} development potential with {months_supply:.1f} months of supply indicating {'undersupplied' if months_supply < 3 else 'balanced'} market.

📍 **Market Demand Analysis**
- Active Inventory: {active_listings:,} units
- Supply-Demand Balance: {months_supply:.1f} months supply
- Market Absorption: {'Fast' if months_supply < 3 else 'Moderate' if months_supply < 6 else 'Slow'}

💹 **Development Feasibility**
{'High demand market - consider accelerated timeline' if months_supply < 3 else 'Moderate demand - standard development timeline recommended'}

🛣️ **Implementation Strategy**
1. Focus on {'move-in ready units' if months_supply < 3 else 'premium features to differentiate'}
2. Target {'entry-level and mid-market' if months_supply < 3 else 'luxury segment'}
3. Plan for {'6-12 month' if months_supply < 3 else '12-18 month'} absorption"""
            }
        ]
        
        return examples
    
    def _estimate_rental_yield(self, median_price: float) -> float:
        """Estimate rental yield based on market price"""
        # Simplified estimation - in real implementation, use actual rental data
        if median_price < 200000:
            return 8.5
        elif median_price < 400000:
            return 7.0
        elif median_price < 600000:
            return 5.5
        else:
            return 4.0
    
    def _estimate_cap_rate(self, median_price: float) -> float:
        """Estimate cap rate based on market conditions"""
        return self._estimate_rental_yield(median_price) - 1.5
    
    def _estimate_cash_on_cash(self, median_price: float) -> float:
        """Estimate cash-on-cash return"""
        return self._estimate_rental_yield(median_price) + 2.0

# Usage
preprocessor = RealEstateDataPreprocessor()
training_data = preprocessor.prepare_training_data(market_df, property_listings)

# Save for fine-tuning
with open('real_estate_training_data.jsonl', 'w') as f:
    for example in training_data:
        f.write(json.dumps(example) + '\n')
```

#### **B. Fine-Tuning Pipeline**

```python
# fine_tuning.py
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from datasets import Dataset
import json

class RealEstateLLMFineTuner:
    """
    Fine-tune LLM for real estate domain expertise
    """
    
    def __init__(self, base_model: str = "microsoft/DialoGPT-medium"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Add special tokens for real estate domain
        special_tokens = {
            "additional_special_tokens": [
                "[INVESTOR]", "[DEVELOPER]", "[BUYER]", "[AGENT]",
                "[MARKET_DATA]", "[PROPERTY]", "[ANALYSIS]", "[RECOMMENDATION]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for fine-tuning
        """
        def tokenize_function(examples):
            # Format: [ROLE] input [ANALYSIS] output
            inputs = []
            for i, example in enumerate(examples['input']):
                role = self._detect_role(example)
                formatted_input = f"[{role.upper()}] {example} [ANALYSIS]"
                full_text = formatted_input + " " + examples['output'][i]
                inputs.append(full_text)
            
            tokenized = self.tokenizer(
                inputs,
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        dataset = Dataset.from_list(training_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def fine_tune(self, training_data: List[Dict[str, str]], 
                  output_dir: str = "./real_estate_llm"):
        """
        Fine-tune the model on real estate data
        """
        dataset = self.prepare_dataset(training_data)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=5e-5,
            warmup_steps=100,
            logging_steps=100,
            logging_dir=f"{output_dir}/logs",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        
        return trainer
    
    def _detect_role(self, query: str) -> str:
        """Detect user role from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["invest", "roi", "cash flow", "rental"]):
            return "investor"
        elif any(word in query_lower for word in ["develop", "zoning", "permit", "construction"]):
            return "developer"
        elif any(word in query_lower for word in ["buy", "home", "family", "neighborhood"]):
            return "buyer"
        elif any(word in query_lower for word in ["list", "market", "client", "lead"]):
            return "agent"
        else:
            return "general"

# Usage
fine_tuner = RealEstateLLMFineTuner()
trainer = fine_tuner.fine_tune(training_data)
```

### **1.2 Enhanced RAG System**

#### **A. Multi-Modal RAG Pipeline**

```python
# enhanced_rag.py
import asyncio
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

class SearchMode(Enum):
    VECTOR_ONLY = "vector"
    GRAPH_ONLY = "graph"
    HYBRID = "hybrid"
    LLM_ONLY = "llm"
    MULTI_MODAL = "multi_modal"

class EnhancedRAGPipeline:
    """
    Advanced RAG pipeline combining multiple search strategies
    with specialized real estate LLM
    """
    
    def __init__(self):
        self.vector_search = None  # Your existing vector search
        self.graph_search = None   # Your existing graph search
        self.market_llm = None     # Fine-tuned real estate LLM
        self.hallucination_detector = HallucinationDetector()
        self.query_router = IntelligentQueryRouter()
        self.context_manager = ContextManager()
        
    async def process_query(self, query: str, user_context: Dict[str, Any], 
                          conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Main query processing pipeline
        """
        # Step 1: Analyze query and determine strategy
        query_analysis = await self.query_router.analyze_query(
            query, user_context, conversation_history
        )
        
        # Step 2: Execute multi-modal search
        search_results = await self._execute_multi_modal_search(
            query, query_analysis, user_context
        )
        
        # Step 3: Generate response using specialized LLM
        response = await self._generate_response(
            query, search_results, query_analysis, user_context
        )
        
        # Step 4: Validate and enhance response
        validated_response = await self._validate_and_enhance(
            response, search_results, query_analysis
        )
        
        # Step 5: Update context and memory
        await self.context_manager.update_context(
            query, validated_response, user_context
        )
        
        return validated_response
    
    async def _execute_multi_modal_search(self, query: str, 
                                        query_analysis: Dict[str, Any],
                                        user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute searches across multiple modalities
        """
        search_tasks = []
        
        # Determine which searches to run based on query analysis
        if query_analysis["needs_factual_data"]:
            search_tasks.append(self.graph_search.search(query))
        
        if query_analysis["needs_semantic_understanding"]:
            search_tasks.append(self.vector_search.search(query))
        
        if query_analysis["needs_market_expertise"]:
            search_tasks.append(self._llm_knowledge_search(query, user_context))
        
        if query_analysis["needs_real_time_data"]:
            search_tasks.append(self._external_api_search(query, user_context))
        
        # Execute searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and rank results
        combined_results = self._combine_search_results(
            search_results, query_analysis
        )
        
        return combined_results
    
    async def _llm_knowledge_search(self, query: str, 
                                  user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use fine-tuned LLM's internal knowledge for domain-specific insights
        """
        role = user_context.get("role", "general")
        
        # Create role-specific prompt
        role_prompt = self._get_role_prompt(role)
        
        # Query the fine-tuned model for domain knowledge
        llm_response = await self.market_llm.generate(
            f"{role_prompt}\n\nQuery: {query}\n\nProvide domain-specific insights:"
        )
        
        return [{
            "content": llm_response,
            "source": "domain_llm",
            "relevance_score": 0.9,
            "result_type": "domain_knowledge"
        }]
    
    async def _external_api_search(self, query: str, 
                                 user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch real-time data from external APIs
        """
        entities = await self._extract_entities(query)
        api_results = []
        
        # Search for current listings if properties mentioned
        if entities.get("properties") or entities.get("locations"):
            listings = await self._fetch_current_listings(entities)
            api_results.extend(listings)
        
        # Fetch market data if market analysis needed
        if entities.get("locations"):
            market_data = await self._fetch_market_data(entities["locations"])
            api_results.extend(market_data)
        
        return api_results
    
    async def _generate_response(self, query: str, search_results: Dict[str, Any],
                               query_analysis: Dict[str, Any], 
                               user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using specialized LLM with RAG context
        """
        role = user_context.get("role", "general")
        
        # Build context from search results
        context = self._build_context(search_results, query_analysis)
        
        # Create role-specific prompt with context
        prompt = self._create_contextualized_prompt(
            query, context, role, user_context
        )
        
        # Generate response using fine-tuned model
        response = await self.market_llm.generate(
            prompt,
            max_tokens=1500,
            temperature=0.7,
            top_p=0.9
        )
        
        return {
            "response": response,
            "context_used": context,
            "confidence_score": self._calculate_confidence(response, context),
            "sources": self._extract_sources(search_results)
        }
    
    async def _validate_and_enhance(self, response: Dict[str, Any],
                                  search_results: Dict[str, Any],
                                  query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate response and enhance with additional information
        """
        # Check for hallucinations
        hallucination_score = await self.hallucination_detector.detect(
            response["response"], search_results
        )
        
        # Enhance with data visualizations if needed
        if query_analysis["needs_visualization"]:
            response["visualizations"] = await self._generate_visualizations(
                search_results, query_analysis
            )
        
        # Add actionable next steps
        response["next_steps"] = await self._generate_next_steps(
            response["response"], query_analysis
        )
        
        # Add confidence and validation scores
        response["validation"] = {
            "hallucination_score": hallucination_score,
            "confidence_score": response["confidence_score"],
            "data_freshness": self._assess_data_freshness(search_results)
        }
        
        return response
```

#### **B. Hallucination Detection System**

```python
# hallucination_detection.py
from typing import List, Dict, Any, Tuple
import re
import json

class RealEstateHallucinationDetector:
    """
    Specialized hallucination detection for real estate domain
    """
    
    def __init__(self):
        self.factual_patterns = {
            "prices": r'\$[\d,]+',
            "percentages": r'\d+\.?\d*%',
            "dates": r'\d{4}|\w+ \d{4}',
            "addresses": r'\d+\s+[A-Z][a-z]+\s+(?:St|Ave|Rd|Dr|Way|Ln)',
            "sqft": r'\d+,?\d*\s*(?:sq|square)\s*(?:ft|feet)',
            "bedrooms": r'\d+\s*(?:BR|bed|bedroom)',
            "bathrooms": r'\d+\.?\d*\s*(?:BA|bath|bathroom)'
        }
        
        self.real_estate_entities = {
            "locations": ["Austin", "Dallas", "Houston", "Travis County", "Harris County"],
            "property_types": ["Single Family", "Condo", "Townhouse", "Multi-family"],
            "metrics": ["ROI", "Cap Rate", "Cash Flow", "Appreciation"]
        }
    
    async def detect_hallucinations(self, response: str, 
                                  search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential hallucinations in real estate responses
        """
        detection_results = {
            "overall_score": 0.0,
            "factual_accuracy": 0.0,
            "data_grounding": 0.0,
            "entity_consistency": 0.0,
            "issues": []
        }
        
        # Check factual accuracy
        factual_score, factual_issues = self._check_factual_accuracy(
            response, search_results
        )
        
        # Check data grounding
        grounding_score, grounding_issues = self._check_data_grounding(
            response, search_results
        )
        
        # Check entity consistency
        entity_score, entity_issues = self._check_entity_consistency(response)
        
        # Calculate overall score
        detection_results.update({
            "factual_accuracy": factual_score,
            "data_grounding": grounding_score,
            "entity_consistency": entity_score,
            "overall_score": (factual_score + grounding_score + entity_score) / 3,
            "issues": factual_issues + grounding_issues + entity_issues
        })
        
        return detection_results
    
    def _check_factual_accuracy(self, response: str, 
                              search_results: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Check if factual claims in response are supported by search results
        """
        issues = []
        total_claims = 0
        supported_claims = 0
        
        # Extract factual claims from response
        for pattern_name, pattern in self.factual_patterns.items():
            claims = re.findall(pattern, response)
            total_claims += len(claims)
            
            for claim in claims:
                if self._is_claim_supported(claim, search_results, pattern_name):
                    supported_claims += 1
                else:
                    issues.append(f"Unsupported {pattern_name} claim: {claim}")
        
        accuracy_score = supported_claims / total_claims if total_claims > 0 else 1.0
        
        return accuracy_score, issues
    
    def _check_data_grounding(self, response: str, 
                            search_results: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Check if response is properly grounded in retrieved data
        """
        issues = []
        
        # Check for specific data citations
        has_citations = bool(re.search(r'based on|according to|data shows', response.lower()))
        
        # Check for vague claims without data support
        vague_patterns = [
            r'(?:typically|usually|generally|often)\s+(?:properties|markets|investments)',
            r'(?:most|many|some)\s+(?:investors|buyers|agents)',
            r'(?:good|bad|excellent|poor)\s+(?:investment|opportunity|market)'
        ]
        
        vague_claims = sum(len(re.findall(pattern, response.lower())) 
                          for pattern in vague_patterns)
        
        # Calculate grounding score
        grounding_score = 1.0
        if not has_citations and vague_claims > 2:
            grounding_score -= 0.3
            issues.append("Response contains vague claims without specific data support")
        
        if not search_results or len(search_results.get("results", [])) == 0:
            grounding_score -= 0.4
            issues.append("Response generated without sufficient search results")
        
        return max(0.0, grounding_score), issues
    
    def _check_entity_consistency(self, response: str) -> Tuple[float, List[str]]:
        """
        Check for consistent use of real estate entities and terminology
        """
        issues = []
        consistency_score = 1.0
        
        # Check for mixed location references
        mentioned_locations = []
        for location in self.real_estate_entities["locations"]:
            if location.lower() in response.lower():
                mentioned_locations.append(location)
        
        # Check for conflicting property type references
        mentioned_types = []
        for prop_type in self.real_estate_entities["property_types"]:
            if prop_type.lower() in response.lower():
                mentioned_types.append(prop_type)
        
        # Detect inconsistencies
        if len(mentioned_locations) > 3:
            consistency_score -= 0.2
            issues.append("Response mentions too many different locations")
        
        if len(mentioned_types) > 2:
            consistency_score -= 0.1
            issues.append("Response mentions multiple conflicting property types")
        
        return max(0.0, consistency_score), issues
    
    def _is_claim_supported(self, claim: str, search_results: Dict[str, Any], 
                          claim_type: str) -> bool:
        """
        Check if a specific claim is supported by search results
        """
        if not search_results or "results" not in search_results:
            return False
        
        # Convert search results to text for matching
        results_text = " ".join([
            result.get("content", "") for result in search_results["results"]
        ])
        
        # Simple containment check - in production, use more sophisticated matching
        return claim.lower() in results_text.lower()
```

## **Phase 2: Integration Layer (Weeks 5-8)**

### **2.1 Intelligent Query Router**

```python
# intelligent_router.py
from typing import Dict, Any, List
import asyncio
from dataclasses import dataclass
from enum import Enum

@dataclass
class QueryAnalysis:
    intent: str
    entities: Dict[str, List[str]]
    complexity: str
    data_requirements: List[str]
    user_role: str
    confidence: float

class IntelligentQueryRouter:
    """
    Advanced query routing with multi-strategy decision making
    """
    
    def __init__(self):
        self.entity_extractor = RealEstateEntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.complexity_analyzer = ComplexityAnalyzer()
        
    async def analyze_query(self, query: str, user_context: Dict[str, Any],
                          conversation_history: List[Dict] = None) -> QueryAnalysis:
        """
        Comprehensive query analysis for routing decisions
        """
        # Run analysis components in parallel
        entities_task = self.entity_extractor.extract(query)
        intent_task = self.intent_classifier.classify(query, user_context)
        complexity_task = self.complexity_analyzer.analyze(query)
        
        entities, intent, complexity = await asyncio.gather(
            entities_task, intent_task, complexity_task
        )
        
        # Determine data requirements
        data_requirements = self._determine_data_requirements(
            intent, entities, complexity, user_context
        )
        
        # Calculate confidence
        confidence = self._calculate_routing_confidence(
            intent, entities, complexity
        )
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            complexity=complexity,
            data_requirements=data_requirements,
            user_role=user_context.get("role", "general"),
            confidence=confidence
        )
    
    def _determine_data_requirements(self, intent: str, entities: Dict[str, List],
                                   complexity: str, user_context: Dict[str, Any]) -> List[str]:
        """
        Determine what data sources are needed for the query
        """
        requirements = []
        
        # Base requirements by intent
        intent_requirements = {
            "factual_lookup": ["graph_search", "external_apis"],
            "market_analysis": ["vector_search", "graph_search", "llm_knowledge"],
            "investment_analysis": ["vector_search", "graph_search", "llm_knowledge", "external_apis"],
            "property_search": ["graph_search", "external_apis"],
            "comparative_analysis": ["vector_search", "graph_search", "llm_knowledge"]
        }
        
        requirements.extend(intent_requirements.get(intent, ["vector_search"]))
        
        # Additional requirements based on entities
        if entities.get("locations"):
            requirements.append("location_data")
        
        if entities.get("properties"):
            requirements.append("property_details")
        
        if entities.get("agents"):
            requirements.append("agent_information")
        
        # Role-specific requirements
        role_requirements = {
            "investor": ["financial_modeling", "market_trends"],
            "developer": ["zoning_data", "permits_data"],
            "buyer": ["property_listings", "neighborhood_data"],
            "agent": ["market_intelligence", "lead_data"]
        }
        
        user_role = user_context.get("role", "general")
        requirements.extend(role_requirements.get(user_role, []))
        
        return list(set(requirements))  # Remove duplicates
    
    def _calculate_routing_confidence(self, intent: str, entities: Dict[str, List],
                                    complexity: str) -> float:
        """
        Calculate confidence in routing decision
        """
        base_confidence = 0.7
        
        # Boost confidence for clear intent
        intent_confidence_boost = {
            "factual_lookup": 0.2,
            "property_search": 0.15,
            "market_analysis": 0.1,
            "investment_analysis": 0.1,
            "comparative_analysis": 0.05
        }
        
        confidence = base_confidence + intent_confidence_boost.get(intent, 0)
        
        # Adjust for entity clarity
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        if total_entities > 0:
            confidence += min(0.1, total_entities * 0.02)
        
        # Adjust for complexity
        complexity_adjustment = {
            "simple": 0.1,
            "moderate": 0.0,
            "complex": -0.1
        }
        
        confidence += complexity_adjustment.get(complexity, 0)
        
        return min(1.0, max(0.1, confidence))

class ComplexityAnalyzer:
    """
    Analyze query complexity for routing decisions
    """
    
    def __init__(self):
        self.complexity_indicators = {
            "simple": [
                r"what is", r"how much", r"when", r"where",
                r"price of", r"address", r"phone number"
            ],
            "moderate": [
                r"compare", r"analyze", r"recommend", r"should I",
                r"best", r"better", r"pros and cons"
            ],
            "complex": [
                r"develop a strategy", r"comprehensive analysis",
                r"investment portfolio", r"market forecast",
                r"risk assessment", r"feasibility study"
            ]
        }
    
    async def analyze(self, query: str) -> str:
        """
        Analyze query complexity
        """
        query_lower = query.lower()
        scores = {"simple": 0, "moderate": 0, "complex": 0}
        
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[complexity] += 1
        
        # Additional complexity factors
        if len(query.split()) > 20:
            scores["complex"] += 1
        
        if len(re.findall(r'\?', query)) > 1:
            scores["complex"] += 1
        
        return max(scores.items(), key=lambda x: x[1])[0]
```

### **2.2 Context Management System**

```python
# context_management.py
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class ContextManager:
    """
    Manage conversation context and user memory
    """
    
    def __init__(self):
        self.user_contexts = {}  # In production, use Redis or similar
        self.conversation_histories = {}
        
    async def get_context(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user context
        """
        return self.user_contexts.get(user_id, self._create_default_context())
    
    async def update_context(self, user_id: str, query: str, 
                           response: Dict[str, Any], 
                           additional_context: Dict[str, Any] = None):
        """
        Update user context based on interaction
        """
        context = await self.get_context(user_id)
        
        # Update conversation history
        self._add_to_conversation_history(user_id, query, response)
        
        # Extract and update user preferences
        self._update_user_preferences(context, query, response)
        
        # Update search patterns
        self._update_search_patterns(context, query, response)
        
        # Merge additional context
        if additional_context:
            context.update(additional_context)
        
        # Update timestamp
        context["last_interaction"] = datetime.utcnow().isoformat()
        
        # Save context
        self.user_contexts[user_id] = context
    
    def _create_default_context(self) -> Dict[str, Any]:
        """
        Create default user context
        """
        return {
            "role": "general",
            "preferences": {
                "budget_range": None,
                "preferred_locations": [],
                "property_types": [],
                "investment_strategy": None
            },
            "search_history": [],
            "interaction_patterns": {
                "preferred_response_style": "balanced",
                "detail_level": "moderate",
                "technical_level": "intermediate"
            },
            "goals": [],
            "constraints": [],
            "created_at": datetime.utcnow().isoformat(),
            "last_interaction": datetime.utcnow().isoformat()
        }
    
    def _add_to_conversation_history(self, user_id: str, query: str, 
                                   response: Dict[str, Any]):
        """
        Add interaction to conversation history
        """
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response.get("response", ""),
            "confidence": response.get("validation", {}).get("confidence_score", 0),
            "sources_used": response.get("sources", [])
        }
        
        self.conversation_histories[user_id].append(interaction)
        
        # Keep only last 50 interactions
        if len(self.conversation_histories[user_id]) > 50:
            self.conversation_histories[user_id] = self.conversation_histories[user_id][-50:]
    
    def _update_user_preferences(self, context: Dict[str, Any], 
                               query: str, response: Dict[str, Any]):
        """
        Extract and update user preferences from interactions
        """
        query_lower = query.lower()
        
        # Extract budget information
        budget_pattern = r'\$[\d,]+(?:\s*k|\s*thousand|\s*million)?'
        budget_matches = re.findall(budget_pattern, query_lower)
        if budget_matches:
            # Simple budget extraction - in production, use more sophisticated parsing
            context["preferences"]["budget_range"] = budget_matches[0]
        
        # Extract location preferences
        # This would use your entity extractor
        locations = self._extract_locations_from_query(query)
        if locations:
            existing_locations = context["preferences"]["preferred_locations"]
            for location in locations:
                if location not in existing_locations:
                    existing_locations.append(location)
        
        # Extract property type preferences
        property_types = ["single family", "condo", "townhouse", "multi-family"]
        for prop_type in property_types:
            if prop_type in query_lower:
                if prop_type not in context["preferences"]["property_types"]:
                    context["preferences"]["property_types"].append(prop_type)
    
    def _update_search_patterns(self, context: Dict[str, Any], 
                              query: str, response: Dict[str, Any]):
        """
        Update user search patterns for personalization
        """
        # Track search topics
        search_entry = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "response_satisfaction": response.get("validation", {}).get("confidence_score", 0)
        }
        
        context["search_history"].append(search_entry)
        
        # Keep only last 100 searches
        if len(context["search_history"]) > 100:
            context["search_history"] = context["search_history"][-100:]
    
    def _extract_locations_from_query(self, query: str) -> List[str]:
        """
        Simple location extraction - replace with your entity extractor
        """
        # This is a simplified version
        common_locations = ["Austin", "Dallas", "Houston", "San Antonio", "Travis County"]
        found_locations = []
        
        for location in common_locations:
            if location.lower() in query.lower():
                found_locations.append(location)
        
        return found_locations

    async def get_conversation_context(self, user_id: str, 
                                     last_n_interactions: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for maintaining conversation flow
        """
        if user_id not in self.conversation_histories:
            return []
        
        return self.conversation_histories[user_id][-last_n_interactions:]
```

## **Phase 3: Role-Specific Agents (Weeks 9-12)**

### **3.1 Investor Agent Implementation**

```python
# investor_agent.py
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np

class InvestorAgent:
    """
    Specialized agent for real estate investment analysis and advice
    """
    
    def __init__(self, rag_pipeline, market_llm, financial_calculator):
        self.rag_pipeline = rag_pipeline
        self.market_llm = market_llm
        self.financial_calculator = financial_calculator
        self.role_prompt = self._load_investor_prompt()
        
    async def process_investment_query(self, query: str, 
                                     user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process investment-related queries with specialized analysis
        """
        # Analyze investment intent
        investment_intent = await self._analyze_investment_intent(query)
        
        # Gather relevant data
        data_package = await self._gather_investment_data(query, investment_intent)
        
        # Perform financial analysis
        financial_analysis = await self._perform_financial_analysis(
            data_package, investment_intent, user_context
        )
        
        # Generate investment advice
        advice = await self._generate_investment_advice(
            query, data_package, financial_analysis, user_context
        )
        
        # Create action plan
        action_plan = await self._create_action_plan(
            advice, investment_intent, user_context
        )
        
        return {
            "analysis": financial_analysis,
            "advice": advice,
            "action_plan": action_plan,
            "risk_assessment": financial_analysis.get("risk_assessment"),
            "confidence": self._calculate_advice_confidence(data_package, financial_analysis)
        }
    
    async def _analyze_investment_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze specific investment intent from query
        """
        investment_intents = {
            "market_analysis": ["market", "trends", "analysis", "overview"],
            "property_evaluation": ["property", "deal", "analysis", "worth it"],
            "roi_calculation": ["roi", "return", "cash flow", "profit"],
            "market_comparison": ["compare", "vs", "versus", "better"],
            "portfolio_advice": ["portfolio", "diversify", "strategy"],
            "timing_advice": ["when", "timing", "now", "wait"]
        }
        
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, keywords in investment_intents.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "primary_intent": primary_intent,
            "intent_confidence": intent_scores[primary_intent] / len(investment_intents[primary_intent]),
            "secondary_intents": [intent for intent, score in intent_scores.items() 
                                if score > 0 and intent != primary_intent]
        }
    
    async def _gather_investment_data(self, query: str, 
                                    investment_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather all data needed for investment analysis
        """
        # Use RAG pipeline to get relevant data
        rag_results = await self.rag_pipeline.process_query(
            query, {"role": "investor"}, []
        )
        
        # Extract specific financial data
        financial_data = await self._extract_financial_data(rag_results)
        
        # Get market comparables
        market_data = await self._get_market_comparables(query, rag_results)
        
        # Historical performance data
        historical_data = await self._get_historical_performance(query, rag_results)
        
        return {
            "rag_results": rag_results,
            "financial_data": financial_data,
            "market_data": market_data,
            "historical_data": historical_data,
            "data_quality_score": self._assess_data_quality(rag_results)
        }
    
    async def _perform_financial_analysis(self, data_package: Dict[str, Any],
                                        investment_intent: Dict[str, Any],
                                        user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive financial analysis
        """
        financial_data = data_package["financial_data"]
        market_data = data_package["market_data"]
        
        analysis = {}
        
        # ROI Analysis
        if investment_intent["primary_intent"] in ["roi_calculation", "property_evaluation"]:
            analysis["roi_analysis"] = await self.financial_calculator.calculate_roi(
                financial_data, market_data, user_context
            )
        
        # Cash Flow Analysis
        analysis["cash_flow_analysis"] = await self.financial_calculator.calculate_cash_flow(
            financial_data, user_context
        )
        
        # Market Trend Analysis
        analysis["market_trends"] = await self._analyze_market_trends(
            market_data, data_package["historical_data"]
        )
        
        # Risk Assessment
        analysis["risk_assessment"] = await self._assess_investment_risk(
            financial_data, market_data, data_package["historical_data"]
        )
        
        # Comparative Analysis
        if investment_intent["primary_intent"] == "market_comparison":
            analysis["comparative_analysis"] = await self._perform_comparative_analysis(
                data_package, user_context
            )
        
        return analysis
    
    async def _generate_investment_advice(self, query: str, 
                                        data_package: Dict[str, Any],
                                        financial_analysis: Dict[str, Any],
                                        user_context: Dict[str, Any]) -> str:
        """
        Generate personalized investment advice using specialized LLM
        """
        # Build context for LLM
        context = {
            "user_budget": user_context.get("preferences", {}).get("budget_range"),
            "investment_goals": user_context.get("goals", []),
            "risk_tolerance": user_context.get("preferences", {}).get("risk_tolerance", "moderate"),
            "financial_analysis": financial_analysis,
            "market_data": data_package["market_data"]
        }
        
        # Create specialized prompt
        prompt = f"""
        {self.role_prompt}
        
        INVESTMENT CONTEXT:
        - User Budget: {context["user_budget"]}
        - Investment Goals: {', '.join(context["investment_goals"])}
        - Risk Tolerance: {context["risk_tolerance"]}
        
        FINANCIAL ANALYSIS:
        {json.dumps(financial_analysis, indent=2)}
        
        MARKET DATA:
        {json.dumps(data_package["market_data"], indent=2)}
        
        USER QUERY: {query}
        
        Provide specific, actionable investment advice following the investor response format.
        """
        
        advice = await self.market_llm.generate(
            prompt,
            max_tokens=1500,
            temperature=0.3  # Lower temperature for more conservative advice
        )
        
        return advice
    
    async def _create_action_plan(self, advice: str, 
                                investment_intent: Dict[str, Any],
                                user_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create specific action plan based on advice
        """
        action_plan = []
        
        # Base actions for all investment advice
        action_plan.extend([
            {
                "action": "Review Financial Position",
                "description": "Assess current financial situation and investment capacity",
                "timeline": "This week",
                "priority": "High"
            },
            {
                "action": "Get Pre-approval",
                "description": "Obtain financing pre-approval if needed",
                "timeline": "Next 2 weeks",
                "priority": "High"
            }
        ])
        
        # Intent-specific actions
        if investment_intent["primary_intent"] == "property_evaluation":
            action_plan.extend([
                {
                    "action": "Property Inspection",
                    "description": "Schedule professional property inspection",
                    "timeline": "Before making offer",
                    "priority": "Critical"
                },
                {
                    "action": "Market Analysis",
                    "description": "Verify comparables and market conditions",
                    "timeline": "This week",
                    "priority": "High"
                }
            ])
        
        elif investment_intent["primary_intent"] == "market_analysis":
            action_plan.extend([
                {
                    "action": "Market Research",
                    "description": "Deep dive into target market fundamentals",
                    "timeline": "Next 2 weeks",
                    "priority": "Medium"
                },
                {
                    "action": "Network Building",
                    "description": "Connect with local agents and property managers",
                    "timeline": "Ongoing",
                    "priority": "Medium"
                }
            ])
        
        return action_plan
    
    def _load_investor_prompt(self) -> str:
        """
        Load specialized investor prompt template
        """
        return """
        You are a Real Estate Investment Advisor AI specializing in data-driven investment strategies. 
        Your expertise covers residential and commercial real estate across all US markets.

        RESPONSE FORMAT:
        🎯 **Investment Snapshot** (key metrics and immediate takeaway)
        📊 **Market Analysis** (current conditions, trends, forecasts)
        💰 **Financial Projections** (ROI, cash flow, appreciation estimates)
        ⚠️ **Risk Assessment** (potential challenges and mitigation strategies)
        🎯 **Investment Strategy** (recommended approach and timing)
        📋 **Next Steps** (specific actionable items)

        Always provide specific numbers, cite data sources, and give actionable recommendations.
        """
```

### **3.2 Integration Testing Framework**

```python
# integration_testing.py
import asyncio
import pytest
from typing import Dict, Any, List

class EndToEndTestSuite:
    """
    Comprehensive testing suite for the integrated RAG + LLM system
    """
    
    def __init__(self, system):
        self.system = system
        self.test_scenarios = self._load_test_scenarios()
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """
        Run complete test suite
        """
        results = {
            "component_tests": await self._test_components(),
            "integration_tests": await self._test_integration(),
            "end_to_end_tests": await self._test_end_to_end(),
            "performance_tests": await self._test_performance(),
            "accuracy_tests": await self._test_accuracy()
        }
        
        return results
    
    async def _test_components(self) -> Dict[str, bool]:
        """
        Test individual components
        """
        return {
            "vector_search": await self._test_vector_search(),
            "graph_search": await self._test_graph_search(),
            "llm_generation": await self._test_llm_generation(),
            "hallucination_detection": await self._test_hallucination_detection(),
            "query_routing": await self._test_query_routing()
        }
    
    async def _test_end_to_end(self) -> Dict[str, Any]:
        """
        Test complete user journeys
        """
        test_results = {}
        
        for scenario in self.test_scenarios:
            try:
                result = await self.system.process_query(
                    scenario["query"],
                    scenario["user_context"],
                    scenario.get("conversation_history", [])
                )
                
                test_results[scenario["name"]] = {
                    "success": True,
                    "response_quality": self._evaluate_response_quality(
                        result, scenario["expected_elements"]
                    ),
                    "response_time": result.get("processing_time", 0),
                    "confidence": result.get("validation", {}).get("confidence_score", 0)
                }
                
            except Exception as e:
                test_results[scenario["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        return test_results
    
    def _load_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        Load comprehensive test scenarios
        """
        return [
            {
                "name": "investor_market_analysis",
                "query": "I want to invest $500K in Austin real estate. What's the market like?",
                "user_context": {"role": "investor", "budget": "$500K"},
                "expected_elements": [
                    "market analysis", "ROI calculations", "investment strategy", 
                    "risk assessment", "specific recommendations"
                ]
            },
            {
                "name": "buyer_home_search",
                "query": "Find me a 3-bedroom home under $400K in Travis County",
                "user_context": {"role": "buyer", "budget": "$400K"},
                "expected_elements": [
                    "property recommendations", "neighborhood analysis", 
                    "buying strategy", "next steps"
                ]
            },
            {
                "name": "agent_market_intelligence",
                "query": "What's the best marketing strategy for luxury homes in West Austin?",
                "user_context": {"role": "agent", "specialty": "luxury"},
                "expected_elements": [
                    "market intelligence", "marketing recommendations", 
                    "pricing strategy", "competitive analysis"
                ]
            },
            {
                "name": "developer_feasibility",
                "query": "Is a 200-unit apartment complex viable in North Austin?",
                "user_context": {"role": "developer", "project_type": "multifamily"},
                "expected_elements": [
                    "feasibility analysis", "market demand", "regulatory requirements", 
                    "financial projections", "timeline"
                ]
            }
        ]
    
    def _evaluate_response_quality(self, response: Dict[str, Any], 
                                 expected_elements: List[str]) -> float:
        """
        Evaluate response quality against expected elements
        """
        response_text = response.get("response", "").lower()
        
        elements_found = sum(1 for element in expected_elements 
                           if element.lower() in response_text)
        
        quality_score = elements_found / len(expected_elements)
        
        # Bonus for proper formatting
        if "🎯" in response.get("response", ""):
            quality_score += 0.1
        
        # Penalty for hallucination
        hallucination_score = response.get("validation", {}).get("hallucination_score", 1.0)
        if hallucination_score < 0.8:
            quality_score -= 0.2
        
        return min(1.0, max(0.0, quality_score))
```

## **Phase 4: Deployment and Monitoring (Weeks 13-16)**

### **4.1 AWS Deployment Configuration**

```python
# aws_deployment.py
import boto3
from typing import Dict, Any
import json

class AWSDeploymentManager:
    """
    Manage AWS deployment of the complete RAG + LLM system
    """
    
    def __init__(self):
        self.sagemaker = boto3.client('sagemaker')
        self.lambda_client = boto3.client('lambda')
        self.api_gateway = boto3.client('apigateway')
        
    def deploy_complete_system(self) -> Dict[str, str]:
        """
        Deploy the complete system to AWS
        """
        deployment_config = {}
        
        # Deploy fine-tuned LLM models
        deployment_config["llm_endpoints"] = self._deploy_llm_models()
        
        # Deploy RAG pipeline
        deployment_config["rag_pipeline"] = self._deploy_rag_pipeline()
        
        # Deploy API Gateway
        deployment_config["api_gateway"] = self._deploy_api_gateway()
        
        # Deploy monitoring
        deployment_config["monitoring"] = self._deploy_monitoring()
        
        return deployment_config
    
    def _deploy_llm_models(self) -> Dict[str, str]:
        """
        Deploy fine-tuned LLM models to SageMaker
        """
        model_configs = {
            "investor_model": {
                "model_name": "trackrealties-investor-llm",
                "model_data": "s3://trackrealties-models/investor-model.tar.gz",
                "instance_type": "ml.g4dn.xlarge"
            },
            "developer_model": {
                "model_name": "trackrealties-developer-llm", 
                "model_data": "s3://trackrealties-models/developer-model.tar.gz",
                "instance_type": "ml.g4dn.xlarge"
            },
            "buyer_model": {
                "model_name": "trackrealties-buyer-llm",
                "model_data": "s3://trackrealties-models/buyer-model.tar.gz", 
                "instance_type": "ml.g4dn.xlarge"
            },
            "agent_model": {
                "model_name": "trackrealties-agent-llm",
                "model_data": "s3://trackrealties-models/agent-model.tar.gz",
                "instance_type": "ml.g4dn.xlarge"
            }
        }
        
        deployed_endpoints = {}
        
        for role, config in model_configs.items():
            endpoint_name = self._create_sagemaker_endpoint(config)
            deployed_endpoints[role] = endpoint_name
        
        return deployed_endpoints
    
    def _create_sagemaker_endpoint(self, config: Dict[str, str]) -> str:
        """
        Create SageMaker endpoint for model
        """
        # Create model
        model_response = self.sagemaker.create_model(
            ModelName=config["model_name"],
            PrimaryContainer={
                'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04',
                'ModelDataUrl': config["model_data"],
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                    'TRANSFORMERS_CACHE': '/opt/ml/model'
                }
            },
            ExecutionRoleArn='arn:aws:iam::YOUR-ACCOUNT:role/SageMakerExecutionRole'
        )
        
        # Create endpoint configuration
        endpoint_config_name = f"{config['model_name']}-config"
        self.sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': config["model_name"],
                    'InitialInstanceCount': 1,
                    'InstanceType': config["instance_type"],
                    'InitialVariantWeight': 1
                }
            ]
        )
        
        # Create endpoint
        endpoint_name = f"{config['model_name']}-endpoint"
        self.sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        return endpoint_name
```

### **4.2 Performance Monitoring**

```python
# monitoring.py
import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

class SystemMonitor:
    """
    Monitor system performance and accuracy
    """
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.metrics_namespace = 'TrackRealties/RAG'
        
    async def monitor_system_health(self) -> Dict[str, Any]:
        """
        Monitor overall system health
        """
        return {
            "response_times": await self._monitor_response_times(),
            "accuracy_metrics": await self._monitor_accuracy(),
            "user_satisfaction": await self._monitor_user_satisfaction(),
            "error_rates": await self._monitor_error_rates(),
            "resource_utilization": await self._monitor_resources()
        }
    
    async def _monitor_response_times(self) -> Dict[str, float]:
        """
        Monitor response times across components
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        metrics = {}
        
        components = ['VectorSearch', 'GraphSearch', 'LLMGeneration', 'OverallPipeline']
        
        for component in components:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=self.metrics_namespace,
                MetricName=f'{component}ResponseTime',
                Dimensions=[],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                metrics[component] = {
                    'average': response['Datapoints'][0]['Average'],
                    'maximum': response['Datapoints'][0]['Maximum']
                }
        
        return metrics
    
    async def _monitor_accuracy(self) -> Dict[str, float]:
        """
        Monitor accuracy metrics
        """
        # This would integrate with your validation system
        return {
            "hallucination_rate": 0.05,  # 5% hallucination rate
            "factual_accuracy": 0.95,    # 95% factual accuracy
            "user_satisfaction": 4.2,    # 4.2/5 average rating
            "query_success_rate": 0.92   # 92% successful query resolution
        }
    
    def publish_metrics(self, metrics: Dict[str, Any]):
        """
        Publish custom metrics to CloudWatch
        """
        for metric_name, value in metrics.items():
            self.cloudwatch.put_metric_data(
                Namespace=self.metrics_namespace,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': 'Count' if 'count' in metric_name.lower() else 'Seconds',
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
```

## **Implementation Timeline Summary**

| Phase | Duration | Key Deliverables | Team Members |
|-------|----------|------------------|--------------|
| **Phase 1: Foundation** | 4 weeks | Fine-tuned LLM, Enhanced RAG, Hallucination Detection | ML Engineer, Data Scientist |
| **Phase 2: Integration** | 4 weeks | Query Router, Context Manager, API Integration | Backend Engineer, ML Engineer |
| **Phase 3: Agents** | 4 weeks | Role-specific Agents, Testing Framework | Full Stack Engineer, QA Engineer |
| **Phase 4: Deployment** | 4 weeks | AWS Deployment, Monitoring, Production Ready | DevOps Engineer, ML Engineer |

## **Success Metrics**

### **Technical Metrics**
- Response Time: < 2 seconds end-to-end
- Accuracy: > 95% factual correctness
- Hallucination Rate: < 5%
- Uptime: > 99.5%

### **Business Metrics**
- User Engagement: > 4.5/5 satisfaction
- Query Success Rate: > 90%
- Conversion to Action: > 60%
- User Retention: > 80% monthly

This implementation guide provides a complete roadmap for building an industry-standard RAG + Real Estate LLM system that any ML engineer can follow step-by-step.