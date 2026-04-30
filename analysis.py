def calculate_average(numbers): 
    """Calculate average of a list""" 
    if not numbers: 
        return 0 
    return sum(numbers) / len(numbers) 
 
def find_max(numbers): 
    """Find maximum value""" 
    return max(numbers) if numbers else None 
