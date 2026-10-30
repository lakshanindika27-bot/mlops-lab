provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "kind-mlops-lab"
}

variable "replicas" {
  type    = number
  default = 1
}

resource "kubernetes_deployment_v1" "api" {
  metadata {
    name = "mlops-api-tf"
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = { app = "mlops-api-tf" }
    }

    template {
      metadata {
        labels = { app = "mlops-api-tf" }
      }

      spec {
        container {
          name              = "api"
          image             = "mlops-lab:v5"
          image_pull_policy = "IfNotPresent"

          port {
            container_port = 8000
          }

          env_from {
            secret_ref {
              name = "mlflow-creds"
            }
          }

          env {
            name  = "MODEL_URI"
            value = "models:/newsgroups-classifier@production"
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "api" {
  metadata {
    name = "mlops-api-tf"
  }

  spec {
    selector = { app = "mlops-api-tf" }

    port {
      name        = "http"
      port        = 80
      target_port = 8000
    }
  }
}
