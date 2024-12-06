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

        lib = pkgs.lib;

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
        devShells.default = pkgs.mkShell (
          let
            pre-commit-bin = "${lib.getBin pkgs.pre-commit}/bin/pre-commit";
          in
          {
            packages =
              [
                python-pkgs
              ]
              ++ (with pkgs; [
                black
                mdformat
                pre-commit
                ruff
                scons
                shfmt
                toml-sort
                treefmt2
                yamlfmt
              ]);

            shellHook = ''
              ${pre-commit-bin} install --allow-missing-config > /dev/null
            '';
          }
        );

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
