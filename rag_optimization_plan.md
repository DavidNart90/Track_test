# TrackRealties RAG Optimization Implementation Plan

## **Executive Summary**

This implementation plan addresses critical issues in the TrackRealties RAG system by establishing intelligent search routing, fixing graph schema mismatches, and optimizing search strategies for real estate domain-specific queries.

## **Current Issues Analysis**

### 🚨 **Critical Problems**
1. **Graph Schema Mismatch**: Queries looking for "Market" nodes vs. actual "MarketData" nodes
2. **Property Name Conflicts**: Searching for "summary" vs. actual "content" properties  
3. **Entity Extraction Disconnect**: Location entities not mapping to graph structure
4. **Missing Search Intelligence**: No logic for when to use Vector vs. Graph vs. Hybrid
5. **Empty Results**: Graph queries returning nodes but no content

## **Phase 1: Search Strategy Intelligence (Priority 1)**

### **1.1 Query Classification Engine**

Create an intelligent query classifier to determine optimal search strategy:

```python
class QueryClassifier:
    """
    Determines optimal search strategy based on query analysis
    """
    
    VECTOR_INDICATORS = [
        "similar", "compare", "like", "analysis", "overview", 
        "recommendations", "insights", "trends", "market conditions"
    ]
    
    GRAPH_INDICATORS = [
        "relationship", "connected", "related", "history", 
        "agent", "office", "who", "when", "which agent"
    ]
    
    HYBRID_INDICATORS = [
        "best", "recommend", "should I", "investment", 
        "roi", "cash flow", "market analysis", "property analysis"
    ]
    
    FACTUAL_INDICATORS = [
        "what is", "how much", "price", "median", "average",
        "inventory", "count", "number", "specific", "exact"
    ]
```

### **1.2 Search Decision Matrix**

| Query Type | Vector DB | Graph DB | Hybrid | Use Case |
|------------|-----------|----------|--------|----------|
| **Factual Data** | ❌ | ✅ | ❌ | "What is median price in Dallas?" |
| **Semantic Analysis** | ✅ | ❌ | ❌ | "Tell me about market trends" |
| **Relationship Queries** | ❌ | ✅ | ❌ | "Who is the listing agent for this property?" |
| **Investment Analysis** | ❌ | ❌ | ✅ | "Should I invest in Austin real estate?" |
| **Comparative Analysis** | ✅ | ✅ | ✅ | "Compare Austin vs Dallas markets" |
| **Property Recommendations** | ❌ | ❌ | ✅ | "Find me good rental properties" |

## **Phase 2: Graph Schema Alignment (Priority 1)**

### **2.1 Fix Graph Query Alignment**

**Current Issues:**
```cypher
-- ❌ WRONG: Looking for non-existent labels/properties
MATCH (l:Location)<-[:LOCATED_IN]-(m:Market)
RETURN m.summary AS content
```

**Fixed Implementation:**
```cypher
-- ✅ CORRECT: Using actual schema
MATCH (l:Location)<-[:LOCATED_IN]-(p:Property)
MATCH (r:Region)-[:HAS_MARKET_DATA]->(md:MarketData)
WHERE l.city = $city AND l.state = $state
RETURN md.content AS content, md.market_data_id AS id
```

### **2.2 Domain-Specific Graph Queries**

Create specialized queries for real estate domain:

```python
REAL_ESTATE_QUERIES = {
    "market_metrics": """
        MATCH (r:Region {region_id: $region_id})-[:HAS_MARKET_DATA]->(md:MarketData)
        WHERE md.date >= $start_date
        RETURN md.median_price, md.inventory_count, md.days_on_market, md.content
        ORDER BY md.date DESC LIMIT 5
    """,
    
    "property_details": """
        MATCH (p:Property {property_id: $property_id})
        OPTIONAL MATCH (p)-[:LISTED_BY]->(a:Agent)
        OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
        RETURN p.content, a.name, a.phone, l.city, l.state
    """,
    
    "agent_properties": """
        MATCH (a:Agent {agent_id: $agent_id})<-[:LISTED_BY]-(p:Property)
        RETURN p.address, p.price, p.status, p.content
        ORDER BY p.listed_date DESC LIMIT 10
    """,
    
    "location_analysis": """
        MATCH (l:Location {city: $city, state: $state})<-[:LOCATED_IN]-(p:Property)
        RETURN 
            COUNT(p) as total_properties,
            AVG(p.price) as avg_price,
            AVG(p.days_on_market) as avg_dom,
            COLLECT(p.content)[0..3] as sample_properties
    """
}
```

## **Phase 3: Enhanced Entity Extraction (Priority 2)**

### **3.1 Real Estate Entity Extractor**

```python
class RealEstateEntityExtractor:
    """
    Domain-specific entity extraction for real estate queries
    """
    
    def __init__(self):
        self.location_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})',  # Austin, TX
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+([A-Z]{2})',   # Austin TX
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+metro',        # Austin metro
        ]
        
        self.property_patterns = [
            r'(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',          # 123 Main St
            r'property\s+id[:\s]+([a-zA-Z0-9\-]+)',             # property id: abc-123
        ]
        
        self.metric_patterns = [
            r'(median\s+price|inventory\s+count|days\s+on\s+market)',
            r'(roi|cash\s+flow|cap\s+rate|appreciation)',
        ]
    
    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {
            "locations": self._extract_locations(query),
            "properties": self._extract_properties(query),
            "metrics": self._extract_metrics(query),
            "agents": self._extract_agents(query)
        }
        return entities
```

### **3.2 Query Intent Classification**

```python
class QueryIntentClassifier:
    """
    Classifies user intent to determine search strategy
    """
    
    INTENT_PATTERNS = {
        "factual_lookup": [
            r"what\s+is\s+the\s+(median|average|current)",
            r"how\s+much\s+(does|is|are)",
            r"(price|cost|value)\s+of",
        ],
        
        "comparative_analysis": [
            r"compare\s+(\w+)\s+(to|vs|versus|and)",
            r"difference\s+between",
            r"better\s+(investment|buy|choice)",
        ],
        
        "relationship_query": [
            r"who\s+(is|are)\s+the\s+(agent|broker)",
            r"which\s+(agent|office|company)",
            r"(agent|broker)\s+(for|of)\s+this",
        ],
        
        "investment_analysis": [
            r"should\s+I\s+(buy|invest|purchase)",
            r"(roi|return|cash\s+flow|investment)",
            r"(profitable|worth\s+it|good\s+deal)",
        ]
    }
```

## **Phase 4: Smart Search Router (Priority 1)**

### **4.1 Intelligent Search Router Implementation**

```python
class SmartSearchRouter:
    """
    Routes queries to optimal search strategy based on content analysis
    """
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.intent_classifier = QueryIntentClassifier()
        self.entity_extractor = RealEstateEntityExtractor()
    
    async def route_search(self, query: str, user_context: dict = None) -> SearchStrategy:
        """
        Determines optimal search strategy for the given query
        """
        # Extract entities and classify intent
        entities = await self.entity_extractor.extract_entities(query)
        intent = await self.intent_classifier.classify_intent(query)
        query_type = self.query_classifier.classify_query(query)
        
        # Decision logic
        if intent == "factual_lookup" and entities["locations"]:
            return SearchStrategy.GRAPH_ONLY
            
        elif intent == "comparative_analysis":
            return SearchStrategy.HYBRID
            
        elif intent == "relationship_query":
            return SearchStrategy.GRAPH_ONLY
            
        elif intent == "investment_analysis":
            return SearchStrategy.HYBRID
            
        elif query_type == "semantic_analysis":
            return SearchStrategy.VECTOR_ONLY
            
        else:
            return SearchStrategy.HYBRID
    
    async def execute_search(self, query: str, strategy: SearchStrategy) -> List[SearchResult]:
        """
        Execute the determined search strategy
        """
        if strategy == SearchStrategy.VECTOR_ONLY:
            return await self.vector_search.search(query)
            
        elif strategy == SearchStrategy.GRAPH_ONLY:
            return await self.graph_search.search(query)
            
        elif strategy == SearchStrategy.HYBRID:
            return await self.hybrid_search.search(query)
```

## **Phase 5: Implementation Tasks Breakdown**

### **Sprint 1: Critical Fixes (Week 1-2)**

#### **Task 1.1: Fix Graph Schema Alignment** 
- **Priority**: 🔴 Critical
- **Effort**: 2 days
- **Owner**: Backend Developer

**Subtasks:**
1. Audit current graph schema vs. query expectations
2. Update all graph queries to match actual node labels and properties
3. Fix property name mismatches (summary → content)
4. Test graph queries with actual data

**Acceptance Criteria:**
- [ ] All graph queries return actual content, not "No Content"
- [ ] No more Neo4j warnings about unknown labels/properties
- [ ] Graph search returns relevant results for location queries

#### **Task 1.2: Implement Query Classification Engine**
- **Priority**: 🔴 Critical  
- **Effort**: 3 days
- **Owner**: ML Engineer

**Subtasks:**
1. Create QueryClassifier class with intent detection
2. Implement real estate domain-specific patterns
3. Add unit tests for classification accuracy
4. Integrate with existing search pipeline

**Acceptance Criteria:**
- [ ] Correctly classifies 90%+ of test queries
- [ ] Routes factual queries to graph search
- [ ] Routes semantic queries to vector search
- [ ] Routes complex queries to hybrid search

#### **Task 1.3: Enhanced Entity Extraction**
- **Priority**: 🟡 High
- **Effort**: 2 days  
- **Owner**: NLP Engineer

**Subtasks:**
1. Replace basic location extraction with regex patterns
2. Add property ID, agent name, and metric extraction
3. Improve location normalization (Austin, TX vs Austin TX)
4. Test extraction accuracy on real user queries

**Acceptance Criteria:**
- [ ] Extracts locations with 95%+ accuracy
- [ ] Handles various location formats consistently
- [ ] Extracts property IDs and agent names correctly

### **Sprint 2: Smart Routing (Week 3-4)**

#### **Task 2.1: Implement Smart Search Router**
- **Priority**: 🟡 High
- **Effort**: 4 days
- **Owner**: Backend Developer

**Subtasks:**
1. Create SmartSearchRouter class
2. Implement search strategy decision logic
3. Add fallback mechanisms for edge cases
4. Integrate with existing RAG pipeline

**Acceptance Criteria:**
- [ ] Routes queries to appropriate search method
- [ ] Handles edge cases gracefully
- [ ] Improves search result relevance by 40%+

#### **Task 2.2: Optimize Graph Queries for Real Estate Domain**
- **Priority**: 🟡 High
- **Effort**: 3 days
- **Owner**: Graph Database Developer

**Subtasks:**
1. Create domain-specific query templates
2. Optimize query performance with proper indexing
3. Add caching for frequently accessed market data
4. Implement query result ranking

**Acceptance Criteria:**
- [ ] Graph queries execute in <500ms
- [ ] Results ranked by relevance to query
- [ ] Proper handling of location-based queries

### **Sprint 3: Advanced Features (Week 5-6)**

#### **Task 3.1: Hybrid Search Optimization**
- **Priority**: 🟢 Medium
- **Effort**: 3 days
- **Owner**: ML Engineer

**Subtasks:**
1. Implement intelligent result fusion
2. Add query-specific weight adjustments
3. Optimize vector-graph result combination
4. Add result diversity scoring

#### **Task 3.2: Performance Monitoring & Analytics**
- **Priority**: 🟢 Medium
- **Effort**: 2 days
- **Owner**: DevOps Engineer

**Subtasks:**
1. Add search strategy performance metrics
2. Implement A/B testing framework
3. Create search analytics dashboard
4. Monitor query classification accuracy

## **Phase 6: Success Metrics & Monitoring**

### **6.1 Key Performance Indicators**

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **Search Accuracy** | ~60% | 90%+ | User satisfaction scores |
| **Query Classification** | N/A | 95%+ | Manual validation |
| **Response Time** | ~3s | <1s | API response times |
| **Graph Query Success** | ~20% | 95%+ | Non-empty results |
| **User Engagement** | N/A | +40% | Session duration |

### **6.2 Monitoring Implementation**

```python
class SearchAnalytics:
    """
    Monitors and analyzes search performance
    """
    
    async def log_search_execution(self, query: str, strategy: SearchStrategy, 
                                 results: List[SearchResult], response_time: float):
        """
        Log search execution for analysis
        """
        analytics_data = {
            "timestamp": datetime.utcnow(),
            "query": query,
            "strategy": strategy.value,
            "result_count": len(results),
            "response_time": response_time,
            "has_results": len(results) > 0,
            "user_feedback": None  # To be updated later
        }
        
        await self.analytics_store.log_search(analytics_data)
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate search performance analytics report
        """
        return {
            "strategy_performance": await self._analyze_strategy_performance(),
            "query_patterns": await self._analyze_query_patterns(),
            "failure_analysis": await self._analyze_failed_searches(),
            "recommendations": await self._generate_optimization_recommendations()
        }
```

## **Phase 7: Testing Strategy**

### **7.1 Unit Testing**

```python
class TestSearchRouter:
    """
    Unit tests for search routing logic
    """
    
    def test_factual_query_routing(self):
        query = "What is the median price in Dallas, TX?"
        strategy = self.router.classify_query(query)
        assert strategy == SearchStrategy.GRAPH_ONLY
    
    def test_investment_analysis_routing(self):
        query = "Should I invest in Austin real estate?"
        strategy = self.router.classify_query(query)
        assert strategy == SearchStrategy.HYBRID
    
    def test_entity_extraction(self):
        query = "Find properties in Austin, TX with good ROI"
        entities = self.extractor.extract_entities(query)
        assert "Austin" in entities["locations"]
        assert "TX" in entities["locations"]
```

### **7.2 Integration Testing**

1. **End-to-End Search Flow Testing**
2. **Graph Database Connection Testing** 
3. **Vector Database Performance Testing**
4. **Real User Query Testing**

## **Implementation Timeline**

| Phase | Duration | Start Date | Dependencies | Risk Level |
|-------|----------|------------|--------------|------------|
| **Phase 1: Critical Fixes** | 2 weeks | Week 1 | None | 🔴 High |
| **Phase 2: Smart Routing** | 2 weeks | Week 3 | Phase 1 | 🟡 Medium |
| **Phase 3: Advanced Features** | 2 weeks | Week 5 | Phase 2 | 🟢 Low |
| **Phase 4: Monitoring** | 1 week | Week 7 | Phase 3 | 🟢 Low |

## **Risk Mitigation**

### **High-Risk Items**
1. **Graph Schema Changes**: May require data migration
   - **Mitigation**: Test thoroughly in staging environment
   
2. **Search Performance**: Complex routing may slow responses
   - **Mitigation**: Implement caching and async processing
   
3. **Classification Accuracy**: Poor routing decisions hurt UX
   - **Mitigation**: A/B testing and fallback mechanisms

### **Success Criteria**

✅ **Technical Success:**
- Graph queries return relevant, non-empty results
- Query classification accuracy >95%
- Search response time <1 second
- Zero Neo4j schema warnings

✅ **Business Success:**
- User engagement increases 40%+
- Search satisfaction scores >4.5/5
- Successful query resolution >90%
- Support ticket reduction by 30%

## **Next Steps**

1. **Immediate (This Week)**: Fix graph schema alignment issues
2. **Short-term (Next 2 weeks)**: Implement query classification and smart routing
3. **Medium-term (Next month)**: Deploy enhanced entity extraction and optimization
4. **Long-term (Next quarter)**: Full analytics and continuous improvement framework

This plan transforms your RAG system from a basic implementation to an intelligent, domain-optimized real estate assistant that provides accurate, relevant results through smart search strategy selection.