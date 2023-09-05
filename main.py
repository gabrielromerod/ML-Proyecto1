import os

def create_species_folders():
    for i in range(1, 11):
        especie_carpeta = f"Especie_{i}"
        try:
            os.mkdir(especie_carpeta)
        except FileExistsError:
            pass

def organize_images_by_species(directorio_imagenes):
    archivos = os.listdir(directorio_imagenes)

    for archivo in archivos:
        if archivo.endswith(".png"):
            numero_especie = int(archivo[:3])

            carpeta_destino = f"Especie_{numero_especie}"

            origen = os.path.join(directorio_imagenes, archivo)
            destino = os.path.join(carpeta_destino, archivo)

            with open(origen, 'rb') as archivo_origen, open(destino, 'wb') as archivo_destino:
                contenido = archivo_origen.read()
                archivo_destino.write(contenido)

def main():
    directorio_imagenes = "images"
    create_species_folders()
    organize_images_by_species(directorio_imagenes)

if __name__ == "__main__":
    main()

