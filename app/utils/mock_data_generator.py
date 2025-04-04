"""
Mock data generator for fire brigade incident reports.
Creates realistic incident reports with timestamps.
"""
import random
import datetime
import uuid

# Sample incidents with sentiment categories
POSITIVE_REPORTS = [
    "Successfully rescued family from burning building with no injuries.",
    "Quick response led to complete fire containment with minimal damage.",
    "Team effectively managed chemical spill with zero environmental impact.",
    "Saved three cats from a tree without incident.",
    "Rescued elderly resident from flood waters, receiving community praise.",
    "Fire extinguished before spreading to neighboring properties.",
    "Training exercise completed with excellent team coordination.",
    "All personnel returned safely after dangerous rescue operation.",
    "New equipment performed exceptionally well during emergency response.",
    "Community expressed gratitude for rapid response to gas leak."
]

NEUTRAL_REPORTS = [
    "Responded to fire alarm which was determined to be a false alarm.",
    "Conducted routine equipment inspection and maintenance.",
    "Attended monthly safety briefing with all staff.",
    "Dispatched to reported smoke, found controlled barbecue.",
    "Inspected fire safety systems at local business.",
    "Provided standby support at community event.",
    "Meeting held with council regarding disaster planning.",
    "Training session conducted on new protocols.",
    "Quarterly drill completed as scheduled.",
    "Routine patrol of high-risk area conducted."
]

NEGATIVE_REPORTS = [
    "Multiple injuries reported due to building collapse during firefighting operation.",
    "Equipment failure hampered rescue attempts at apartment fire.",
    "Response time delayed due to traffic congestion, increasing property damage.",
    "Volunteer firefighter suffered smoke inhalation requiring hospitalization.",
    "Flood response limited by insufficient resources and personnel.",
    "Communication system failed during critical emergency coordination.",
    "Vehicle collision en route to emergency caused additional injuries.",
    "Budget constraints prevented purchase of essential safety equipment.",
    "High winds spread fire beyond containment lines, destroying additional structures.",
    "Staff shortage led to inadequate coverage during major incident."
]

def generate_random_incident():
    """
    Generate a random incident report with timestamp and ID.
    
    Returns:
        dict: A dictionary containing incident details
    """
    # Randomly select sentiment category
    sentiment_category = random.choice(["positive", "neutral", "negative"])
    
    # Select report based on sentiment
    if sentiment_category == "positive":
        report = random.choice(POSITIVE_REPORTS)
    elif sentiment_category == "neutral":
        report = random.choice(NEUTRAL_REPORTS)
    else:
        report = random.choice(NEGATIVE_REPORTS)
    
    # Generate random timestamp within the last 30 days
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=days_ago, hours=hours_ago, minutes=minutes_ago)
    
    # Format timestamp as ISO string
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Generate unique ID
    incident_id = str(uuid.uuid4())[:8]
    
    return {
        "incident_id": incident_id,
        "report": report,
        "timestamp": formatted_timestamp,
        "_true_sentiment": sentiment_category  # Hidden field for validation
    }

def generate_incidents(count=10):
    """
    Generate multiple random incidents.
    
    Args:
        count (int): Number of incidents to generate
        
    Returns:
        list: List of incident dictionaries
    """
    return [generate_random_incident() for _ in range(count)]

if __name__ == "__main__":
    # Test the generator
    print(generate_incidents(3)) 