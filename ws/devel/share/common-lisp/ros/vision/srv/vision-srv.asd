
(cl:in-package :asdf)

(defsystem "vision-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "NewHost" :depends-on ("_package_NewHost"))
    (:file "_package_NewHost" :depends-on ("_package"))
    (:file "TrackPerson" :depends-on ("_package_TrackPerson"))
    (:file "_package_TrackPerson" :depends-on ("_package"))
  ))