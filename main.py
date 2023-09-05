import os

directorio_imagenes = "images"

archivos = os.listdir(directorio_imagenes)

for i in range(1, 11):
    especie_carpeta = f"Especie_{i}"
    try:
        os.mkdir(especie_carpeta)
    except FileExistsError:
        pass

for archivo in archivos:
    if archivo.endswith(".png"):
        numero_especie = int(archivo[:3])
        
        carpeta_destino = f"Especie_{numero_especie}"
        
        origen = os.path.join(directorio_imagenes, archivo)
        destino = os.path.join(carpeta_destino, archivo)
        
        with open(origen, 'rb') as archivo_origen, open(destino, 'wb') as archivo_destino:
            contenido = archivo_origen.read()
            archivo_destino.write(contenido)


