{
  description = "The Cooper Union - ECE 416: Adaptive Algorithms";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };

        python = pkgs.python3;

        python-pkgs = python.withPackages (
          python-pkgs: with python-pkgs; [
            einops
            ipython
            jupyter
            jupytext
            matplotlib
            numpy
            papermill
            scipy
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            [
              python-pkgs
            ]
            ++ (with pkgs; [
              black
              mdformat
              ruff
              scons
              shfmt
              toml-sort
              treefmt2
              yamlfmt
            ]);
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
