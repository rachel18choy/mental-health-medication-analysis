from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    data_loaded: bool = Field(..., description="Whether data is loaded")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    data_records: int = Field(..., description="Number of data records")
    timestamp: str = Field(..., description="Current timestamp")

class DataResponse(BaseModel):
    """Response schema for data endpoint"""
    data: List[Dict[str, Any]]
    total_records: int
    returned_records: int
    offset: int
    limit: Optional[int] = None

class PredictionInput(BaseModel):
    """Input schema for making predictions - 30 fields matching your CSV"""
    # These 30 fields should match your 30 features (excluding target)
    age: int = Field(25, ge=18, le=100, description="Age in years")
    gender: str = Field("Woman", description="Gender identity")
    study_area: str = Field("Psychology", description="Area of study")
    study_years: str = Field("3-4 years", description="Years of tertiary study")
    recognize_psychosis: int = Field(1, ge=0, le=1, description="Recognize psychosis")
    professional_help_helpful: int = Field(1, ge=0, le=1, description="Believe professional help is helpful")
    john_could_snap_out: int = Field(2, ge=1, le=5, description="John could snap out of it")
    john_weakness: int = Field(2, ge=1, le=5, description="John's problem is weakness")
    john_not_real_illness: int = Field(2, ge=1, le=5, description="Not real medical illness")
    john_dangerous: int = Field(2, ge=1, le=5, description="John is dangerous")
    avoid_john: int = Field(2, ge=1, le=5, description="Best to avoid John")
    john_unpredictable: int = Field(2, ge=1, le=5, description="Makes unpredictable")
    not_tell_anyone: int = Field(2, ge=1, le=5, description="Would not tell anyone")
    go_out_weekend: int = Field(4, ge=1, le=5, description="Go out weekend")
    work_on_project: int = Field(4, ge=1, le=5, description="Work on project")
    invite_to_house: int = Field(4, ge=1, le=5, description="Invite to house")
    go_to_johns_house: int = Field(4, ge=1, le=5, description="Go to John's house")
    develop_friendship: int = Field(4, ge=1, le=5, description="Develop friendship")
    ask_harm_thoughts: int = Field(5, ge=1, le=5, description="Ask about harm thoughts")
    listen_restate: int = Field(5, ge=1, le=5, description="Listen and restate")
    convey_hope: int = Field(5, ge=1, le=5, description="Convey hope")
    discuss_professional_options: int = Field(5, ge=1, le=5, description="Discuss options")
    ask_supportive_people: int = Field(5, ge=1, le=5, description="Ask about support")
    ask_suicide_thoughts: int = Field(5, ge=1, le=5, description="Ask suicide thoughts")
    ask_suicide_plan: int = Field(5, ge=1, le=5, description="Ask suicide plan")
    encourage_professional_help: int = Field(5, ge=1, le=5, description="Encourage help")
    acknowledge_frightened: int = Field(5, ge=1, le=5, description="Acknowledge fear")
    convince_false_beliefs: int = Field(1, ge=1, le=5, description="Convince false")
    listen_unreal_experiences: int = Field(4, ge=1, le=5, description="Listen experiences")
    find_reasons_no_help: int = Field(5, ge=1, le=5, description="Find reasons no help")

class PredictionOutput(BaseModel):
    """Output schema for predictions"""
    prediction: int = Field(..., description="0=Not helpful, 1=Helpful")
    probability: float = Field(..., ge=0, le=1, description="Probability of being helpful")
    model_used: str = Field(..., description="Name of the model used")
    confidence: str = Field(..., description="Confidence level")

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    items: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionOutput]
    total_items: int
    helpful_count: int
    not_helpful_count: int
    helpful_percentage: float