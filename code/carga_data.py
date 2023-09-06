import os
import re
from shutil import move

def identify_species(directory="images"):
    species_set = set()
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print(f"El directorio {directory} no existe.")
        return []
    pattern = re.compile(r'00(\d)0+|0(\d{1,2})0+')
    for filename in filenames:
        match = pattern.search(filename)
        if match:
            species_number = match.group(1) if match.group(1) else match.group(2)
            species_set.add(species_number)
    return sorted(list(species_set))

def create_species_folders(species_list, base_directory="images"):
    for species in species_list:
        folder_path = os.path.join(base_directory, f"Especie_{species}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def organize_images_by_species(directory="images"):
    filenames = os.listdir(directory)
    pattern = re.compile(r'00(\d)0+|0(\d{1,2})0+')
    for filename in filenames:
        match = pattern.search(filename)
        if match:
            species_number = match.group(1) if match.group(1) else match.group(2)
            source = os.path.join(directory, filename)
            destination_folder = os.path.join(directory, f"Especie_{species_number}")
            if os.path.exists(destination_folder):
                destination = os.path.join(destination_folder, filename)
                move(source, destination)
            else:
                print(f"La carpeta de destino {destination_folder} no existe.")

if __name__ == "__main__":
    species_list = identify_species()
    create_species_folders(species_list)
    organize_images_by_species()