from invoke import task
from shlex import quote
from colorama import Fore

@task
def start(c):
    """
    Start the project
    """
    c.run("docker-compose up -d --build")

@task
def stop(c):
    """
    Stop the project
    """
    c.run("docker-compose down -v")

@task
def ps(c):
    """
    List containers status
    """
    c.run("docker ps")

@task
def logs(c):
    """
    Tails containers logs
    """
    c.run("docker-compose logs -f --tail=150")

@task
def console(c):
    """
    Access the app container
    """
    c.run("docker-compose exec app bash", pty=not c.power_shell)

@task
def help(c):
    """
    List all commands
    """
    c.run('inv --list')
