from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from ..config import settings

class MemoryService:
    def __init__(self):
        self.memory_dir = "data/memory"
        os.makedirs(self.memory_dir, exist_ok=True)

    def get_conversation_history(
        self,
        conversation_id: str,
        max_exchanges: int
    ) -> List[Dict]:
        """Get conversation history up to max_exchanges."""
        history_file = os.path.join(self.memory_dir, f"{conversation_id}.json")
        
        if not os.path.exists(history_file):
            return []
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        return history[-max_exchanges:]

    def add_to_conversation(
        self,
        conversation_id: str,
        question: str,
        answer: str
    ):
        """Add a Q&A exchange to conversation history."""
        history_file = os.path.join(self.memory_dir, f"{conversation_id}.json")
        
        # Load existing history or create new
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new exchange
        history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only the most recent exchanges
        if len(history) > settings.CONVERSATION_HISTORY_SIZE:
            history = history[-settings.CONVERSATION_HISTORY_SIZE:]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def add_feedback(
        self,
        answer_id: str,
        rating: int,
        comment: Optional[str] = None
    ):
        """Store feedback for an answer."""
        feedback_file = os.path.join(self.memory_dir, "feedback.json")
        
        # Load existing feedback or create new
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback = json.load(f)
        else:
            feedback = []
        
        # Add new feedback
        feedback.append({
            'answer_id': answer_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Save updated feedback
        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)

    def get_feedback_stats(self) -> Dict:
        """Get statistics about feedback."""
        feedback_file = os.path.join(self.memory_dir, "feedback.json")
        
        if not os.path.exists(feedback_file):
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'rating_distribution': {}
            }
        
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
        
        if not feedback:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'rating_distribution': {}
            }
        
        # Calculate statistics
        total_rating = sum(item['rating'] for item in feedback)
        rating_distribution = {}
        
        for item in feedback:
            rating = item['rating']
            rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        return {
            'total_feedback': len(feedback),
            'average_rating': total_rating / len(feedback),
            'rating_distribution': rating_distribution
        } 