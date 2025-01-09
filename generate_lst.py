import os

def genera_lista_file_due_cartelle(folder1, folder2, output_file):
    """
    Genera un file .lst contenente i percorsi relativi dei file di due cartelle,
    associati riga per riga nel formato:
    /nome_prime_folder/file \t /nome_secondo_folder/file

    Args:
        folder1 (str): Percorso della prima cartella.
        folder2 (str): Percorso della seconda cartella.
        output_file (str): Nome del file .lst da generare.
    """
    # Controlla se i percorsi forniti sono directory valide
    if not os.path.isdir(folder1):
        raise ValueError(f"Il percorso {folder1} non è una directory valida.")
    if not os.path.isdir(folder2):
        raise ValueError(f"Il percorso {folder2} non è una directory valida.")

    # Ottieni la lista dei file in entrambe le cartelle, ordinata alfabeticamente
    files_folder1 = sorted(os.listdir(folder1))
    files_folder2 = sorted(os.listdir(folder2))

    # Controlla che le due cartelle abbiano lo stesso numero di file
    if len(files_folder1) != len(files_folder2):
        raise ValueError("Le due cartelle non contengono lo stesso numero di file.")

    # Ottieni la parte comune del percorso da rimuovere
    common_prefix = 'PIDNet/data/loveDa/'

    # Apri il file di output in modalità scrittura
    with open(output_file, 'w') as f:
        # Itera sui file delle due cartelle
        for file1, file2 in zip(files_folder1, files_folder2):
            # Costruisci i percorsi completi dei file
            full_path1 = os.path.join(folder1, file1)
            full_path2 = os.path.join(folder2, file2)

            # Verifica che entrambi siano file (ignora le sottodirectory)
            if os.path.isfile(full_path1) and os.path.isfile(full_path2):
                # Rimuovi la parte comune dal percorso e uniforma il separatore
                relative_path1 = full_path1.replace(common_prefix, '').replace(os.sep, '/')
                relative_path2 = full_path2.replace(common_prefix, '').replace(os.sep, '/')

                # Scrivi nel formato specificato nel file di output
                f.write(f"{relative_path1}\t{relative_path2}\n")

# Esempio di utilizzo
folder1 = 'PIDNet/data/loveDa/Train/Rural/images_png'  # Sostituisci con il percorso della prima cartella
folder2 = 'PIDNet/data/loveDa/Train/Rural/masks_png'  # Sostituisci con il percorso della seconda cartella
output = 'PIDNet/data/list/loveDa/target.lst'  # Nome del file di output
genera_lista_file_due_cartelle(folder1, folder2, output)
