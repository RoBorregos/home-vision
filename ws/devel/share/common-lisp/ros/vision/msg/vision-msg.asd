
(cl:in-package :asdf)

(defsystem "vision-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "img" :depends-on ("_package_img"))
    (:file "_package_img" :depends-on ("_package"))
    (:file "img_list" :depends-on ("_package_img_list"))
    (:file "_package_img_list" :depends-on ("_package"))
    (:file "people_count" :depends-on ("_package_people_count"))
    (:file "_package_people_count" :depends-on ("_package"))
    (:file "target" :depends-on ("_package_target"))
    (:file "_package_target" :depends-on ("_package"))
  ))