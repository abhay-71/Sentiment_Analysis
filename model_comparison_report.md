# Sentiment Analysis Models Comparison Report

## Model Overview

This report compares three sentiment analysis models developed for the fire brigade incident analysis application:

1. **Synthetic Model**: Trained on synthetic templates specific to fire brigade domain
2. **Twitter Model**: Trained on real Twitter data with diverse contexts
3. **Hybrid Model**: A combined approach that leverages both models based on text domain and confidence scores

## Performance Comparison

### Overall Accuracy

| Model | Overall Accuracy | Fire Brigade Domain | General Text |
|-------|------------------|---------------------|--------------|
| Synthetic Model | 76.92% (10/13) | 88.89% (8/9) | 50.00% (2/4) |
| Twitter Model | 38.46% (5/13) | 33.33% (3/9) | 50.00% (2/4) |
| Hybrid Model | 61.54% (8/13) | 77.78% (7/9) | 25.00% (1/4) |

### Performance by Sentiment Type

#### Synthetic Model
- **Positive**: 3/4 correct (75.00%)
- **Neutral**: 3/3 correct (100.00%)
- **Negative**: 4/6 correct (66.67%)

#### Twitter Model
- **Positive**: 3/4 correct (75.00%)
- **Neutral**: 0/3 correct (0.00%)
- **Negative**: 2/6 correct (33.33%)

#### Hybrid Model
- **Positive**: 4/4 correct (100.00%)
- **Neutral**: 1/3 correct (33.33%)
- **Negative**: 3/6 correct (50.00%)

## Strengths and Weaknesses

### Synthetic Model

**Strengths:**
- Excellent performance on domain-specific text (88.89%)
- Perfect performance on neutral sentiment (100%)
- Good overall accuracy (76.92%)
- Strong confidence scores for domain text

**Weaknesses:**
- Limited generalization to non-domain text (50%)
- May not adapt well to novel expressions

### Twitter Model

**Strengths:**
- Good performance on positive sentiment (75%)
- Trained on diverse real-world language
- Better with colloquial expressions
- Equal performance on general text as synthetic model (50%)

**Weaknesses:**
- Poor overall accuracy (38.46%)
- Struggles with domain-specific text (33.33%)
- Completely fails on neutral sentiment (0%)
- Lower confidence scores overall

### Hybrid Model

**Strengths:**
- Perfect performance on positive sentiment (100%)
- Good performance on domain-specific text (77.78%)
- Successfully combines strengths of both models
- Makes intelligent choices based on domain detection
- Improved performance on negative sentiment over Twitter model

**Weaknesses:**
- Struggles with general text (25%)
- Lower accuracy on neutral sentiment (33.33%)
- Overall performance lower than synthetic model alone
- Added complexity in implementation

## Decision Logic Analysis

The hybrid model implements several key decision strategies:

1. **Domain Detection**: Uses keyword analysis to determine if text is fire brigade specific
2. **Confidence Weighting**: Adjusts the influence of each model based on domain specificity
3. **Low Confidence Handling**: Defaults to neutral when both models have low confidence
4. **Non-Neutral Preference**: In case of ties, gives preference to non-neutral predictions

### Model Selection Patterns

In our test examples, the hybrid model made the following choices:
- Used Twitter model for 3 examples (23.1%)
- Used Synthetic model for 5 examples (38.5%) 
- Defaulted to low confidence in 3 examples (23.1%)
- Used the twitter_non_neutral logic in 0 examples (0%)
- Used the synthetic_non_neutral logic in 2 examples (15.4%)

## Examples Highlights

### Where Hybrid Model Improved

Example: "Today's shift was exhausting with multiple emergency calls"
- Twitter model: ✗ Neutral (confidence: 0.1670)
- Synthetic model: ✓ Negative (confidence: 0.2377)
- Hybrid model: ✓ Negative (confidence: 0.2377)
- Logic: synthetic_non_neutral

Example: "Our new thermal imaging cameras are making search and rescue much more effective"
- Twitter model: ✗ Negative (confidence: 0.2760)
- Synthetic model: ✓ Positive (confidence: 0.4073)
- Hybrid model: ✓ Positive (confidence: 0.4073)
- Logic: synthetic (domain detection)

### Where Hybrid Model Failed

Example: "The service was terrible and I will never come back" 
- Twitter model: ✓ Negative (confidence: 0.1630)
- Synthetic model: ✗ Neutral (confidence: 0.1636)
- Hybrid model: ✗ Neutral (confidence: 0.1636) 
- Logic: low_confidence_default

Example: "It's an okay option if you don't have alternatives"
- Twitter model: ✗ Positive (confidence: 0.3106)
- Synthetic model: ✓ Neutral (confidence: 0.1682)
- Hybrid model: ✗ Positive (confidence: 0.3106)
- Logic: twitter (higher confidence score)

## Recommendations

Based on the comparative analysis, we recommend:

1. **Use the Synthetic Model for Production**:
   - Best overall accuracy
   - Excellent performance on domain-specific text
   - Provides the most stable predictions for fire brigade applications

2. **Further Hybrid Model Improvements**:
   - Adjust confidence thresholds to favor synthetic model more heavily
   - Improve domain detection with more sophisticated NLP techniques
   - Fine-tune weighting mechanism based on additional validation data
   - Consider ensemble voting rather than confidence-based selection

3. **Next Steps for Model Development**:
   - Collect domain-specific labeled data for further training
   - Implement active learning to improve from user feedback
   - Consider fine-tuning Twitter model on domain-specific examples
   - Explore transformer-based models like BERT for improved language understanding

## Conclusion

While the hybrid model showed promise by achieving perfect positive sentiment detection and improving over the Twitter model's performance, the synthetic model remains the most reliable choice for the fire brigade application in its current state. The synthetic model's superior performance on domain-specific text (88.89%) makes it the most suitable choice for production use.

The hybrid approach demonstrates the potential for combining models with complementary strengths, but requires further refinement of its decision logic to outperform the synthetic model consistently. For future development, collecting more domain-specific labeled data would likely yield the most significant improvements to any of these approaches. 