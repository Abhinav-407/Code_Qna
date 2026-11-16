import typer

app = typer.Typer()

@app.command()
def review(project_path: str):
    print(f"Reviewing project {project_path}")

@app.command()
def ask(project_path: str):
    print(f"Starting Q&A for project: {project_path}")
if __name__ == "__main__":
    app()