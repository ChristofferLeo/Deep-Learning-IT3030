
def fileParser(filePath):
    config = {'GLOBALS': {}, 'LAYERS': []}
    current_section = None
    layer_attrs = {}  # Initialize outside the loop to accumulate attributes across lines

    with open(filePath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            if line in ['GLOBALS', 'LAYERS']:
                if line == 'LAYERS' and layer_attrs:  # Save previous layer if exists
                    config['LAYERS'].append(layer_attrs)
                    layer_attrs = {}  # Reset for the next layer
                current_section = line
                continue

            # Split line into parts for key-value pairs
            parts = line.split()
            for part in parts:
                if ':' in part:  # It's a key-value pair
                    key, value = part.split(':', 1)  # Split only on the first colon
                    if current_section == 'GLOBALS':
                        config['GLOBALS'][key] = value
                    elif current_section == 'LAYERS':
                        layer_attrs[key] = value
                else:  # Handle single attributes (e.g., "type: softmax")
                    # Assuming single attributes are always type specifications
                    layer_attrs['type'] = part

            # If in LAYERS section and not a type-only line, accumulate attributes
            if current_section == 'LAYERS' and ':' in line:
                config['LAYERS'].append(layer_attrs)
                layer_attrs = {}  # Reset for potentially new layer attributes

    # Add the last layer if not already added
    if layer_attrs:
        config['LAYERS'].append(layer_attrs)

    return config

