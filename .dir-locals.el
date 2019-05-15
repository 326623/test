;; .dir-locals.el for basic c++ project set up

;; ((nil
;;   (helm-make-build-dir . "test/build"))
;;  (c++-mode
;;   (helm-make-arguments . "-j8")
;;   (flycheck-gcc-language-standard . "c++11")))

;; ((nil .
;;       ((cmake-ide-project-dir . "~/Project")
;;        (cmake-ide-build-dir . "~/Project/build")
;;        (cmake-ide-cmake-opts . "-DCMAKE_BUILD_TYPE=Debug")
;;        (helm-make-build-dir . "build")
;;        (helm-make-arguments . "-j7")
;;        (helm-ctest-dir . "~/Project/build/")
;;        )))

;; ((nil . ((eval . (setq
;;                   projectile-project-test-cmd #'helm-ctest
;;                   projectile-project-compilation-cmd #'helm-make-projectile
;;                   projectile-project-compilation-dir "build"
;;                   helm-make-build-dir (projectile-compilation-dir)
;;                   helm-ctest-dir (projectile-compilation-dir)
;;                   ))
;;          (projectile-project-name . "My Cool Project")
;;          (projectile-project-run-cmd . "./run.sh")
;;          (projectile-project-configure-cmd . "cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..")
;;          (helm-make-arguments . "-j7"))))

((nil .
      ((helm-make-build-dir . "./build")
       (helm-make-arguments . "-j8")
       (helm-ctest-dir . "build")
       )))
