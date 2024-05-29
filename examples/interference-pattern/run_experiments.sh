

# Path to the file
ORIGINAL_FILE="./sim_4_channels_normal_pattern.cc"
# Directory containing the file
NEW_NAME="./sim.cc"

# Step 1: Rename the file
mv "$ORIGINAL_FILE" "$NEW_NAME"
echo "Renamed $ORIGINAL_FILE to $NEW_NAME"
# Step 2: Run the desired script
./run_models_normal_pattern.sh
# Step 3: Rename the file back to its original name
mv "$NEW_NAME" "$ORIGINAL_FILE"
echo "Renamed $NEW_NAME back to $ORIGINAL_FILE"


# # Path to the file
# ORIGINAL_FILE="./sim_6_channels_normal_pattern.cc"
# # Directory containing the file
# NEW_NAME="./sim.cc"

# # Step 1: Rename the file
# mv "$ORIGINAL_FILE" "$NEW_NAME"
# echo "Renamed $ORIGINAL_FILE to $NEW_NAME"
# # Step 2: Run the desired script
# ./run_models_normal_pattern_6channels.sh
# # Step 3: Rename the file back to its original name
# mv "$NEW_NAME" "$ORIGINAL_FILE"
# echo "Renamed $NEW_NAME back to $ORIGINAL_FILE"

# # Path to the file
# ORIGINAL_FILE="./sim_8_channels_normal_pattern.cc"
# # Directory containing the file
# NEW_NAME="./sim.cc"

# # Step 1: Rename the file
# mv "$ORIGINAL_FILE" "$NEW_NAME"
# echo "Renamed $ORIGINAL_FILE to $NEW_NAME"
# # Step 2: Run the desired script
# ./run_models_normal_pattern_8channels.sh
# # Step 3: Rename the file back to its original name
# mv "$NEW_NAME" "$ORIGINAL_FILE"
# echo "Renamed $NEW_NAME back to $ORIGINAL_FILE"

# # Path to the file
# ORIGINAL_FILE="./sim_4_channels_inverted_pattern.cc"
# # Directory containing the file

# NEW_NAME="./sim.cc"
# # Step 1: Rename the file
# mv "$ORIGINAL_FILE" "$NEW_NAME"
# echo "Renamed $ORIGINAL_FILE to $NEW_NAME"
# # Step 2: Run the desired script
# ./run_models_normal_inverted_pattern.sh
# # Step 3: Rename the file back to its original name
# mv "$NEW_NAME" "$ORIGINAL_FILE"
# echo "Renamed $NEW_NAME back to $ORIGINAL_FILE"

# # Path to the file
# #ORIGINAL_FILE="./sim_4_channels_inverted_pattern.cc"
# ORIGINAL_FILE="./sim_4_channels_normal_pattern.cc"
# # Directory containing the file
# NEW_NAME="./sim.cc"

# ORG_MYGYM="./mygym.cc"

# TEMP_MYGYM="./mygym_normal.cc"

# NOISE_MYGYM="./mygym_noise.cc"

# # Step 1: Rename the file
# mv "$ORIGINAL_FILE" "$NEW_NAME"
# echo "Renamed $ORIGINAL_FILE to $NEW_NAME"

# mv "$ORG_MYGYM" "$TEMP_MYGYM"
# echo "Renamed $ORG_MYGYM to $TEMP_MYGYM"

# mv "$NOISE_MYGYM" "$ORG_MYGYM"
# echo "Renamed $NOISE_MYGYM to $ORG_MYGYM"

# # Step 2: Run the desired script
# ./run_models_normal_pattern_noise.sh
# # Step 3: Rename the file back to its original name
# mv "$NEW_NAME" "$ORIGINAL_FILE"
# echo "Renamed $NEW_NAME back to $ORIGINAL_FILE"

# # Revert the first rename: move $ORG_MYGYM (currently points to $NOISE_MYGYM) back to $NOISE_MYGYM
# mv "$ORG_MYGYM" "$NOISE_MYGYM"
# echo "Renamed $ORG_MYGYM back to $NOISE_MYGYM"

# # Revert the second rename: move $TEMP_MYGYM back to $ORG_MYGYM
# mv "$TEMP_MYGYM" "$ORG_MYGYM"
# echo "Renamed $TEMP_MYGYM back to $ORG_MYGYM"