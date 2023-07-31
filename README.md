# stelevator

The **stel**lar **ev**olution emul**ator**.

Going up!

## Installation

To use the latest development version of the package, follow this guide. To contribute to and develop the package, see [Development](#development).

Clone the package,

```bash
git clone https://github.com/stelevator/stelevator.git
```

Install the package using `pip`,

```sh
pip install stelevator
```

## Development

Fork the package and then clone it where `<your-username>` is your GitHub username,

```sh
git clone https://github.com/<your-username>/stelevator.git
```

Install the package in your favourite virtual environment. Use the 'editable' flag to ensure live changes are registered with your package manager,

```sh
pip install -e stelevator
```

Add the `stelevator` remote upstream so that you can pull changes from the main development version,

```sh
cd stelevator
git remote add upstream https://github.com/stelevator/stelevator.git
git remote -v
```

which should output something like this,

```
origin	https://github.com/<your-username>/stelevator.git (fetch)
origin	https://github.com/<your-username>/stelevator.git (push)
upstream	https://github.com/stelevator/stelevator.git (fetch)
upstream	https://github.com/stelevator/stelevator.git (push)
```
